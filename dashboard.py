import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score

# ── Configuração da página ──
st.set_page_config(
    page_title="Manutenção Preditiva — AI4I 2020",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Paleta ──
COR = {
    'normal':    '#3B82F6',
    'falha':     '#F59E0B',
    'accent':    '#D97706',
    'perigo':    '#EF4444',
    'roxo':      '#8B5CF6',
    'sucesso':   '#059669',
    'cinza':     '#6B7280',
    'grid':      'rgba(255,255,255,0.07)',
    'texto_sec': '#9CA3AF',
}

NOMES_MODOS = {
    'TWF': 'Tool Wear', 'HDF': 'Heat Dissipation',
    'PWF': 'Power', 'OSF': 'Overstrain', 'RNF': 'Random',
}

CORES_MODO = {
    'TWF': '#F59E0B', 'HDF': '#EF4444',
    'PWF': '#8B5CF6', 'OSF': '#3B82F6', 'RNF': '#6B7280',
}

def plotly_base(**kwargs):
    layout = dict(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=50, b=20),
        font=dict(family='Inter, system-ui, sans-serif', size=13),
        xaxis=dict(showgrid=True, gridcolor=COR['grid'], zeroline=False),
        yaxis=dict(showgrid=True, gridcolor=COR['grid'], zeroline=False),
        legend=dict(bgcolor='rgba(0,0,0,0)'),
    )
    layout.update(kwargs)
    return layout

# ── CSS ──
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    .stApp { font-family: 'Inter', system-ui, sans-serif; }

    /* Sidebar */
    section[data-testid="stSidebar"] { border-right: 1px solid rgba(255,255,255,0.06); }

    /* Metric cards */
    .kpi-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.03) 0%, rgba(255,255,255,0.06) 100%);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 1.1rem 1.3rem;
        text-align: center;
        transition: border-color 0.2s;
    }
    .kpi-card:hover { border-color: rgba(217,119,6,0.5); }
    .kpi-card .kpi-label {
        font-size: 0.72rem; text-transform: uppercase;
        letter-spacing: 0.08em; color: #9CA3AF; margin-bottom: 0.25rem;
    }
    .kpi-card .kpi-value {
        font-size: 1.7rem; font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Tabela de regras */
    .regras-table { width:100%; border-collapse:separate; border-spacing:0; border-radius:10px; overflow:hidden; border:1px solid rgba(255,255,255,0.08); margin:0.8rem 0; }
    .regras-table th { padding:0.75rem 1rem; font-size:0.72rem; text-transform:uppercase; letter-spacing:0.05em; color:#9CA3AF; text-align:left; border-bottom:1px solid rgba(255,255,255,0.08); background:rgba(255,255,255,0.02); }
    .regras-table td { padding:0.65rem 1rem; font-size:0.88rem; border-bottom:1px solid rgba(255,255,255,0.04); }
    .regras-table tr:last-child td { border-bottom:none; }
    .badge { display:inline-block; padding:0.18rem 0.55rem; border-radius:999px; font-size:0.73rem; font-weight:600; }
    .badge.det { background:rgba(5,150,105,0.15); color:#34D399; }
    .badge.prob { background:rgba(217,119,6,0.15); color:#FBBF24; }
    .badge.full { background:rgba(5,150,105,0.15); color:#34D399; }
    .badge.partial { background:rgba(239,68,68,0.15); color:#FCA5A5; }

    /* Insight box */
    .insight-box {
        background: rgba(255,255,255,0.02);
        border-left: 3px solid #D97706;
        border-radius: 0 8px 8px 0;
        padding: 0.9rem 1.1rem;
        margin: 0.8rem 0;
        font-size: 0.88rem; line-height: 1.6;
    }
    .insight-box code {
        background: rgba(255,255,255,0.08);
        padding: 0.12rem 0.35rem; border-radius: 4px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem; color: #D97706;
    }

    /* Monospace nos metrics nativos */
    [data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace !important; }
    hr { border-color: rgba(255,255,255,0.06) !important; }
</style>
""", unsafe_allow_html=True)


# ── Dados ──
@st.cache_data
def carregar_dados():
    try:
        df = pd.read_csv('ai4i2020.csv')
    except FileNotFoundError:
        st.error("Arquivo `ai4i2020.csv` não encontrado. Coloque-o na mesma pasta do script.")
        st.stop()

    df['delta_temp'] = df['Process temperature [K]'] - df['Air temperature [K]']
    df['power_W'] = df['Torque [Nm]'] * df['Rotational speed [rpm]'] * (2 * np.pi / 60)
    df['wear_torque'] = df['Tool wear [min]'] * df['Torque [Nm]']
    df['regra_HDF'] = ((df['delta_temp'] < 8.6) & (df['Rotational speed [rpm]'] < 1380)).astype(int)
    df['regra_PWF'] = ((df['power_W'] < 3500) | (df['power_W'] > 9000)).astype(int)
    th = {'L': 11000, 'M': 12000, 'H': 13000}
    df['threshold_OSF'] = df['Type'].map(th)
    df['regra_OSF'] = (df['wear_torque'] > df['threshold_OSF']).astype(int)
    df['regra_TWF_zona'] = ((df['Tool wear [min]'] >= 200) & (df['Tool wear [min]'] <= 240)).astype(int)
    return df

@st.cache_data
def treinar_modelos(df):
    le = LabelEncoder()
    dm = df.copy()
    dm['Type_encoded'] = le.fit_transform(dm['Type'])
    fb = ['Air temperature [K]','Process temperature [K]','Rotational speed [rpm]',
          'Torque [Nm]','Tool wear [min]','Type_encoded']
    fe = fb + ['delta_temp','power_W','wear_torque']
    modos = ['TWF','HDF','PWF','OSF','RNF']
    y = dm[modos]
    Xb_tr,Xb_te,y_tr,y_te = train_test_split(dm[fb],y,test_size=0.3,random_state=42,stratify=dm['Machine failure'])
    Xe_tr,Xe_te,_,_ = train_test_split(dm[fe],y,test_size=0.3,random_state=42,stratify=dm['Machine failure'])
    cb = MultiOutputClassifier(RandomForestClassifier(n_estimators=200,class_weight='balanced',random_state=42,n_jobs=-1))
    cb.fit(Xb_tr,y_tr); yb = cb.predict(Xb_te)
    ce = MultiOutputClassifier(RandomForestClassifier(n_estimators=200,class_weight='balanced',random_state=42,n_jobs=-1))
    ce.fit(Xe_tr,y_tr); ye = ce.predict(Xe_te)
    res = []
    for i,m in enumerate(modos):
        f1b = f1_score(y_te.iloc[:,i],yb[:,i],zero_division=0)
        f1e = f1_score(y_te.iloc[:,i],ye[:,i],zero_division=0)
        res.append({'Modo':m,'F1_base':round(f1b,2),'F1_eng':round(f1e,2),'delta':round(f1e-f1b,2)})
    imps = {}
    for i,m in enumerate(modos):
        imps[m] = dict(zip(fe, ce.estimators_[i].feature_importances_))
    return pd.DataFrame(res), imps, fe

df = carregar_dados()

# ── Sidebar ──
with st.sidebar:
    st.header("⚙ Filtros")
    tipos_sel = st.multiselect("Tipo de produto", options=['L','M','H'], default=['L','M','H'])
    df_f = df[df['Type'].isin(tipos_sel)] if tipos_sel else df
    st.divider()
    st.caption("**Dataset:** AI4I 2020 Predictive Maintenance")
    st.caption("Matzka, S. (2020) · UCI ML Repository")
    st.caption("DOI: `10.24432/C5HS5C`")

# ── Header ──
st.title("Manutenção Preditiva — AI4I 2020")
st.markdown("Validação de regras documentadas e classificação multi-label de modos de falha.")
st.divider()

tab1, tab2, tab3 = st.tabs(["📊 Visão Geral", "📐 Validação das Regras", "🤖 Classificação Multi-label"])

# ════════════════════════════════════════════════════════════
# TAB 1
# ════════════════════════════════════════════════════════════
with tab1:
    total = len(df_f)
    falhas = int(df_f['Machine failure'].sum())
    taxa = falhas / total * 100 if total > 0 else 0
    modos_l = ['TWF','HDF','PWF','OSF','RNF']
    modo_top = df_f[modos_l].sum().idxmax() if total > 0 else "—"

    c1,c2,c3,c4 = st.columns(4)
    c1.markdown(f'<div class="kpi-card"><div class="kpi-label">Registros</div><div class="kpi-value">{total:,}</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="kpi-card"><div class="kpi-label">Falhas detectadas</div><div class="kpi-value" style="color:#EF4444;">{falhas}</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="kpi-card"><div class="kpi-label">Taxa de falha</div><div class="kpi-value" style="color:#D97706;">{taxa:.1f}%</div></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="kpi-card"><div class="kpi-label">Modo mais frequente</div><div class="kpi-value">{modo_top}</div></div>', unsafe_allow_html=True)

    st.write("")  # spacer

    col_e, col_d = st.columns(2)
    with col_e:
        if total > 0:
            ft = df_f.groupby('Type')['Machine failure'].agg(['sum','count']).reset_index()
            ft['taxa'] = (ft['sum'] / ft['count'] * 100).round(1)
            cores_tipo = {'L':'#3B82F6','M':'#D97706','H':'#059669'}
            fig = px.bar(ft, x='Type', y='taxa', text='taxa', color='Type', color_discrete_map=cores_tipo,
                         title="Taxa de falha por tipo de produto")
            fig.update_traces(texttemplate='%{text}%', textposition='outside', showlegend=False)
            fig.update_layout(**plotly_base(yaxis_title='%', yaxis_range=[0, ft['taxa'].max()*1.4]))
            st.plotly_chart(fig, width="stretch")

    with col_d:
        if total > 0:
            cm = df_f[modos_l].sum().reset_index(); cm.columns = ['Modo','n']
            fig2 = px.bar(cm, x='Modo', y='n', text='n', color='Modo',
                          color_discrete_map=CORES_MODO, title="Ocorrências por modo de falha")
            fig2.update_traces(textposition='outside', showlegend=False)
            fig2.update_layout(**plotly_base(yaxis_title='Qtd', yaxis_range=[0, cm['n'].max()*1.35]))
            st.plotly_chart(fig2, width="stretch")

    fig3 = go.Figure()
    dn = df_f[df_f['Machine failure']==0]; dfl = df_f[df_f['Machine failure']==1]
    fig3.add_trace(go.Scattergl(x=dn['Rotational speed [rpm]'], y=dn['Torque [Nm]'],
        mode='markers', name='Normal', marker=dict(size=3, color=COR['normal'], opacity=0.25)))
    fig3.add_trace(go.Scattergl(x=dfl['Rotational speed [rpm]'], y=dfl['Torque [Nm]'],
        mode='markers', name='Falha', marker=dict(size=6, color=COR['falha'], opacity=0.9, line=dict(width=0.8, color='white'))))
    fig3.update_layout(**plotly_base(
        title='Torque × Velocidade Rotacional', xaxis_title='Rotational speed [rpm]', yaxis_title='Torque [Nm]',
        legend=dict(x=0.85, y=0.95)))
    st.plotly_chart(fig3, width="stretch")


# ════════════════════════════════════════════════════════════
# TAB 2
# ════════════════════════════════════════════════════════════
with tab2:
    st.markdown(
        "Cada modo de falha segue uma regra física descrita na documentação do dataset (Matzka, 2020). "
        "As regras foram reconstruídas em Python e comparadas contra os flags reais."
    )

    regras_val = {
        'HDF': ('regra_HDF','HDF','ΔT &lt; 8.6 K ∧ RPM &lt; 1380'),
        'PWF': ('regra_PWF','PWF','Power &lt; 3500 W ∨ Power &gt; 9000 W'),
        'OSF': ('regra_OSF','OSF','Wear × Torque &gt; limiar (L:11k, M:12k, H:13k)'),
        'TWF': ('regra_TWF_zona','TWF','Tool wear ∈ [200, 240] min'),
    }

    rows_html = ""
    for nome, (col_r, col_f, formula) in regras_val.items():
        total_r = len(df)
        match = (df[col_r] == df[col_f]).sum()
        pct = 100 * match / total_r
        fp = int(((df[col_r]==1)&(df[col_f]==0)).sum())
        fn = int(((df[col_r]==0)&(df[col_f]==1)).sum())
        bt = 'prob' if nome == 'TWF' else 'det'
        tipo_txt = 'Probabilística' if nome == 'TWF' else 'Determinística'
        bc = 'full' if pct == 100 else 'partial'
        cor_modo = CORES_MODO[nome]
        rows_html += f"""<tr>
            <td><span style="color:{cor_modo}; font-weight:600;">{nome}</span>
                <span style="color:#6B7280; font-size:0.8rem;"> {NOMES_MODOS[nome]}</span></td>
            <td><code style="font-size:0.78rem; color:#D97706; background:rgba(255,255,255,0.06); padding:0.12rem 0.35rem; border-radius:4px;">{formula}</code></td>
            <td><span class="badge {bc}">{pct:.1f}%</span></td>
            <td style="font-family:'JetBrains Mono',monospace; font-size:0.85rem;">{fp}</td>
            <td style="font-family:'JetBrains Mono',monospace; font-size:0.85rem;">{fn}</td>
            <td><span class="badge {bt}">{tipo_txt}</span></td></tr>"""

    st.markdown(f"""<table class="regras-table"><thead><tr>
        <th>Modo</th><th>Regra reconstruída</th><th>Concordância</th>
        <th>Falsos +</th><th>Falsos −</th><th>Tipo</th>
    </tr></thead><tbody>{rows_html}</tbody></table>""", unsafe_allow_html=True)

    st.markdown(
        '<div class="insight-box">'
        '<strong>HDF, PWF e OSF</strong> batem 100% — regras determinísticas. '
        '<strong>TWF</strong> tem 747 falsos positivos: a zona 200–240 min contém '
        'substituições preventivas e falhas reais (componente probabilístico). '
        '<strong>RNF</strong> omitido: 0,1% de chance aleatória, sem regra reconstruível.'
        '</div>', unsafe_allow_html=True
    )

    st.write("")
    st.subheader("Exploração visual das regras")
    regra_sel = st.selectbox("Regra:", ["HDF — Dissipação térmica","PWF — Potência","OSF — Sobrecarga"])

    if regra_sel.startswith("HDF"):
        fig_r = go.Figure()
        d0=df[df['HDF']==0]; d1=df[df['HDF']==1]
        fig_r.add_trace(go.Scattergl(x=d0['Rotational speed [rpm]'],y=d0['delta_temp'],mode='markers',name='Normal',
            marker=dict(size=3,color=COR['normal'],opacity=0.25)))
        fig_r.add_trace(go.Scattergl(x=d1['Rotational speed [rpm]'],y=d1['delta_temp'],mode='markers',name='HDF',
            marker=dict(size=6,color=COR['perigo'],opacity=0.9,line=dict(width=0.5,color='#7F1D1D'))))
        fig_r.add_hline(y=8.6,line_dash="dot",line_color=COR['accent'],annotation_text="ΔT = 8.6 K",annotation_font_color=COR['accent'])
        fig_r.add_vline(x=1380,line_dash="dot",line_color=COR['accent'],annotation_text="1380 rpm",annotation_font_color=COR['accent'])
        fig_r.add_shape(type="rect",x0=0,x1=1380,y0=0,y1=8.6,fillcolor="rgba(220,38,38,0.06)",line_width=0)
        fig_r.update_layout(**plotly_base(title='HDF: ΔTemp × Velocidade Rotacional',xaxis_title='RPM',yaxis_title='ΔT (Process − Air) [K]'))
        st.plotly_chart(fig_r, width="stretch")

    elif regra_sel.startswith("PWF"):
        fig_r = go.Figure()
        d0=df[df['PWF']==0]; d1=df[df['PWF']==1]
        fig_r.add_trace(go.Scattergl(x=d0['Rotational speed [rpm]'],y=d0['Torque [Nm]'],mode='markers',name='Normal',
            marker=dict(size=3,color=COR['normal'],opacity=0.25)))
        fig_r.add_trace(go.Scattergl(x=d1['Rotational speed [rpm]'],y=d1['Torque [Nm]'],mode='markers',name='PWF',
            marker=dict(size=6,color=COR['roxo'],opacity=0.9,line=dict(width=0.5,color='#4C1D95'))))
        rpm_r = np.linspace(400,2600,200); rad_s = rpm_r*2*np.pi/60
        fig_r.add_trace(go.Scatter(x=rpm_r,y=3500/rad_s,mode='lines',name='3500 W',line=dict(color=COR['accent'],dash='dot',width=2)))
        fig_r.add_trace(go.Scatter(x=rpm_r,y=9000/rad_s,mode='lines',name='9000 W',line=dict(color=COR['perigo'],dash='dot',width=2)))
        fig_r.update_layout(**plotly_base(title='PWF: Torque × Velocidade (iso-potência)',xaxis_title='RPM',yaxis_title='Torque [Nm]'))
        st.plotly_chart(fig_r, width="stretch")

    elif regra_sel.startswith("OSF"):
        fig_r = go.Figure()
        d0=df[df['OSF']==0]; d1=df[df['OSF']==1]
        fig_r.add_trace(go.Scattergl(x=d0['Tool wear [min]'],y=d0['Torque [Nm]'],mode='markers',name='Normal',
            marker=dict(size=3,color=COR['normal'],opacity=0.25)))
        fig_r.add_trace(go.Scattergl(x=d1['Tool wear [min]'],y=d1['Torque [Nm]'],mode='markers',name='OSF',
            marker=dict(size=6,color='#3B82F6',opacity=0.9,line=dict(width=0.5,color='#1E3A5F'))))
        tw_r = np.linspace(1,300,200)
        for tp,th,cor in [('L',11000,COR['accent']),('M',12000,'#F59E0B'),('H',13000,'#FCD34D')]:
            fig_r.add_trace(go.Scatter(x=tw_r,y=th/tw_r,mode='lines',name=f'{tp} ({th:,})',line=dict(color=cor,dash='dot',width=2)))
        fig_r.update_layout(**plotly_base(title='OSF: Torque × Desgaste (thresholds por tipo)',xaxis_title='Tool wear [min]',yaxis_title='Torque [Nm]',yaxis_range=[0,110]))
        st.plotly_chart(fig_r, width="stretch")


# ════════════════════════════════════════════════════════════
# TAB 3
# ════════════════════════════════════════════════════════════
with tab3:
    st.markdown(
        "Dois modelos Random Forest foram comparados: um com features brutas "
        "e outro com features derivadas das regras documentadas na aba anterior."
    )

    df_res, importances, feat_eng = treinar_modelos(df)

    # Delta cards com st.metric nativo dentro de containers (melhor responsividade)
    st.markdown("#### Ganho de performance (F1-Score)")
    mcols = st.columns(5)
    for i, row in df_res.iterrows():
        with mcols[i]:
            with st.container(border=True):
                delta_txt = f"{row['delta']:+.2f}"
                st.metric(
                    label=f"{row['Modo']} — {NOMES_MODOS[row['Modo']]}",
                    value=f"{row['F1_eng']:.2f}",
                    delta=delta_txt,
                    delta_color="normal"
                )

    st.markdown(
        '<div class="insight-box">'
        '<strong>HDF, PWF e OSF</strong> melhoram significativamente com '
        '<code>delta_temp</code>, <code>power_W</code> e <code>wear_torque</code>, '
        'que capturam os mecanismos exatos de falha. '
        '<strong>TWF</strong> e <strong>RNF</strong> permanecem em F1 = 0.00 '
        '(probabilístico e aleatório, respectivamente).'
        '</div>', unsafe_allow_html=True
    )

    st.write("")

    # F1 comparativo + feature importance lado a lado
    col_g, col_imp = st.columns([1.4, 1])

    with col_g:
        fig_f1 = go.Figure()
        fig_f1.add_trace(go.Bar(
            x=df_res['Modo'], y=df_res['F1_base'], name='Features brutas',
            marker_color=COR['normal'], text=df_res['F1_base'], textposition='outside',
            width=0.35, offset=-0.18
        ))
        fig_f1.add_trace(go.Bar(
            x=df_res['Modo'], y=df_res['F1_eng'], name='Engenheiradas',
            marker_color=COR['accent'], text=df_res['F1_eng'], textposition='outside',
            width=0.35, offset=0.18
        ))
        fig_f1.update_layout(**plotly_base(
            title='F1-Score por modo de falha',
            yaxis_title='F1-Score', yaxis_range=[0, 1.15],
            barmode='group',
            legend=dict(orientation='h', y=1.12, x=0.5, xanchor='center')
        ))
        st.plotly_chart(fig_f1, width="stretch")

    with col_imp:
        st.markdown("#### Importância das features")
        modo_sel = st.selectbox("Modo:", list(NOMES_MODOS.keys()), label_visibility='collapsed')

        rename = {
            'Air temperature [K]':'Air temp', 'Process temperature [K]':'Process temp',
            'Rotational speed [rpm]':'Rot. speed', 'Torque [Nm]':'Torque',
            'Tool wear [min]':'Tool wear', 'Type_encoded':'Product type',
            'delta_temp':'ΔTemp ★', 'power_W':'Power ★', 'wear_torque':'Wear×Torque ★',
        }

        imp_df = pd.DataFrame({
            'Feature': [rename.get(x,x) for x in feat_eng],
            'Importância': list(importances[modo_sel].values())
        }).sort_values('Importância', ascending=True)

        fig_imp = go.Figure(go.Bar(
            y=imp_df['Feature'], x=imp_df['Importância'], orientation='h',
            marker=dict(
                color=imp_df['Importância'],
                colorscale=[[0,'#1F2937'],[0.5,'#D97706'],[1,'#F59E0B']],
            ),
            text=imp_df['Importância'].apply(lambda v: f'{v:.3f}'),
            textposition='outside', textfont=dict(size=11)
        ))
        fig_imp.update_layout(**plotly_base(
            title=f'{NOMES_MODOS[modo_sel]} Failure',
            xaxis_title='Gini importance',
            margin=dict(l=20,r=50,t=50,b=20), height=380
        ))
        st.plotly_chart(fig_imp, width="stretch")
        st.caption("★ = feature derivada das regras físicas documentadas")

# ── Rodapé ──
st.divider()
st.caption("AI4I 2020 Predictive Maintenance Dataset · Matzka (2020) · UCI ML Repository · DOI: 10.24432/C5HS5C")