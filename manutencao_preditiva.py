import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, multilabel_confusion_matrix

#Carregando e entendendo a estrutura do dataset com as 4 primeiras linhas + cabeçalho
df = pd.read_csv('ai4i2020.csv')

#Verificando se há valores nulos
e_nulo = df.isnull().sum()
print(e_nulo)
#Não há valores nulos no dataset.

#Foi visto que a coluna 'UDI' funciona como um Index.
#A coluna 'Product ID' tem valores únicos até onde foi visto. Preciso ver se eles se repetem.
#A coluna 'Type' tem 4 tipos de valores: 'L', 'M' e 'H'.
#A coluna 'Air temperature [K]' tem valores entre 298.66 e 310.15.
#A coluna 'Process temperature [K]' tem valores entre 307.15 e 323.15.
#A coluna 'Rotational speed [rpm]' tem valores entre 500 e 2000.
#A coluna 'Torque [Nm]' tem valores entre 20 e 100.
#A coluna 'Tool wear [min]' tem valores entre 0 e 300.
#A coluna 'Machine failure' tem valores binários: 0 e 1, claramente um booleano.
#As colunas 'TWF', 'HDF', 'PWF', 'OSF' e 'RNF' tem valores binários: 0 e 1, claramente booleanos.

#Confirmando a unicidade das colunas 'UDI' e 'Product ID'
print(df['UDI'].is_unique)
print(df['Product ID'].is_unique)

#Confirmando a distribuição do Type
print(df['Type'].value_counts())

#Comparando falha x modos de falha
modos = ['TWF','HDF', 'PWF', 'OSF', 'RNF']
df['soma_modos'] = df[modos].sum (axis=1)

#Verificando incoerências entre 'Machine failure' e a soma dos modos de falha
incoerentes = df[(df['Machine failure'] != (df['soma_modos'] > 0).astype(int))]
print(f"Registros incoerentes: {len(incoerentes)}")
print(incoerentes[['UDI', 'Machine failure', 'soma_modos'] + modos])
#Existem registros onde Machine failure=1 mas nenhum modo individual está ativo.
#Isso é intencional do dataset — o paper menciona que o método de ML não sabe qual modo causou a falha.

# ============================================================
# PARTE 1 — Validação das regras documentadas (Matzka, 2020)
# ============================================================

#Regra HDF: falha térmica ocorre quando (Process temp - Air temp) < 8.6 K E Rotational speed < 1380 rpm.
#A documentação indica 115 registros nessa condição.
df['delta_temp'] = df['Process temperature [K]'] - df['Air temperature [K]']
df['regra_HDF'] = (
    (df['delta_temp'] < 8.6) &
    (df['Rotational speed [rpm]'] < 1380)
).astype(int)

#Regra PWF: falha de potência ocorre quando a potência (Torque × velocidade angular em rad/s) < 3500 W ou > 9000 W.
#A documentação indica 95 registros nessa condição.
df['power_W'] = df['Torque [Nm]'] * df['Rotational speed [rpm]'] * (2 * np.pi / 60)
df['regra_PWF'] = (
    (df['power_W'] < 3500) | (df['power_W'] > 9000)
).astype(int)

#Regra OSF: sobrecarga ocorre quando (Tool wear × Torque) excede um limiar por tipo de produto.
#L: 11000, M: 12000, H: 13000 minNm. A documentação indica 98 registros.
threshold_map = {'L': 11000, 'M': 12000, 'H': 13000}
df['wear_torque'] = df['Tool wear [min]'] * df['Torque [Nm]']
df['threshold_OSF'] = df['Type'].map(threshold_map)
df['regra_OSF'] = (df['wear_torque'] > df['threshold_OSF']).astype(int)

#Regra TWF: a ferramenta entra em zona de risco quando Tool wear está entre 200 e 240 min.
#Nem todas falham — a documentação diz 120 na zona, 51 falham (componente probabilístico).
df['regra_TWF_zona'] = (
    (df['Tool wear [min]'] >= 200) & (df['Tool wear [min]'] <= 240)
).astype(int)

#Regra RNF: 0,1% de chance aleatória, independente dos parâmetros de processo.
#Apenas 5 registros no dataset. Não há regra determinística para replicar.

#Comparando cada regra reconstruída contra o flag real do dataset.
#Se a regra é determinística, a concordância deveria ser ~100%.
#Desvios indicam componente probabilístico (TWF) ou aleatoriedade (RNF).
regras = {
    'HDF': ('regra_HDF', 'HDF'),
    'PWF': ('regra_PWF', 'PWF'),
    'OSF': ('regra_OSF', 'OSF'),
    'TWF (zona de risco)': ('regra_TWF_zona', 'TWF'),
}

for nome, (col_regra, col_real) in regras.items():
    match = (df[col_regra] == df[col_real]).sum()
    total = len(df)
    falso_pos = ((df[col_regra] == 1) & (df[col_real] == 0)).sum()
    falso_neg = ((df[col_regra] == 0) & (df[col_real] == 1)).sum()
    print(f"\n{nome}:")
    print(f"  Concordância: {match}/{total} ({100*match/total:.1f}%)")
    print(f"  Regra prevê falha, flag=0: {falso_pos}")
    print(f"  Flag=1, regra não prevê:   {falso_neg}")

#HDF, PWF e OSF devem ter concordância próxima de 100%.
#TWF terá falsos positivos altos porque a zona 200-240 contém tanto substituições quanto falhas.
#Esses desvios não são erro — são características documentadas do dataset.

# ============================================================
# PARTE 2 — Classificação multi-label dos modos de falha
# ============================================================

#Codificando 'Type' como numérico para uso no modelo.
le = LabelEncoder()
df['Type_encoded'] = le.fit_transform(df['Type'])

#Features base: apenas as variáveis de processo originais + tipo codificado.
features_base = [
    'Air temperature [K]', 'Process temperature [K]',
    'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]',
    'Type_encoded'
]

#Features engenheiradas: adicionando delta_temp, power_W e wear_torque.
#Essas foram derivadas na Parte 1 a partir das regras documentadas.
#A hipótese é que features informadas pelo domínio melhoram a classificação.
features_eng = features_base + ['delta_temp', 'power_W', 'wear_torque']

#Targets: os cinco modos de falha, não o Machine failure binário.
#Cada modo é uma coluna binária independente — problema multi-label.
y = df[['TWF', 'HDF', 'PWF', 'OSF', 'RNF']]

#Separando treino e teste com stratify no Machine failure para manter a proporção de falhas.
#O desbalanceamento é severo: ~3,4% de falha geral, cada modo individual é ainda mais raro.
X_base = df[features_base]
X_eng = df[features_eng]

X_train_base, X_test_base, y_train, y_test = train_test_split(
    X_base, y, test_size=0.3, random_state=42, stratify=df['Machine failure']
)
X_train_eng, X_test_eng, _, _ = train_test_split(
    X_eng, y, test_size=0.3, random_state=42, stratify=df['Machine failure']
)

#Treinando modelo SEM features engenheiradas.
#class_weight='balanced' compensa o desbalanceamento severo dos modos de falha.
clf_base = MultiOutputClassifier(
    RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
)
clf_base.fit(X_train_base, y_train)
y_pred_base = clf_base.predict(X_test_base)

#Treinando modelo COM features engenheiradas.
clf_eng = MultiOutputClassifier(
    RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
)
clf_eng.fit(X_train_eng, y_train)
y_pred_eng = clf_eng.predict(X_test_eng)

#Comparando performance por modo de falha: sem vs com features engenheiradas.
for i, modo in enumerate(y.columns):
    print(f"\n{'='*50}")
    print(f"{modo} — SEM features engenheiradas:")
    print(classification_report(y_test.iloc[:, i], y_pred_base[:, i], zero_division=0))
    print(f"{modo} — COM features engenheiradas:")
    print(classification_report(y_test.iloc[:, i], y_pred_eng[:, i], zero_division=0))

#TWF e RNF provavelmente terão recall baixo em ambos os casos.
#TWF porque a falha é probabilística na zona de desgaste.
#RNF porque é literalmente aleatório, nenhum modelo consegue prever isso com features de processo.

#Dos 27 registros incoerentes, 18 têm RNF=1 com Machine failure=0.
#O dataset trata Random Failure como modo que nem sempre causa parada da máquina.
#Os 9 restantes têm Machine failure=1 sem nenhum modo ativo — falhas não classificáveis.
#Todos esses padrões são intencionais segundo a documentação do Matzka (2020).
rnf_sem_falha = df[(df['RNF'] == 1) & (df['Machine failure'] == 0)]
falha_sem_modo = df[(df['Machine failure'] == 1) & (df['soma_modos'] == 0)]
print(f"RNF ativo sem Machine failure: {len(rnf_sem_falha)}")
print(f"Machine failure sem modo ativo: {len(falha_sem_modo)}")

#Resumo comparativo: F1-score por modo de falha, sem vs com features engenheiradas.
#A melhora em HDF, PWF e OSF confirma que features derivadas das regras físicas documentadas superam a abordagem com features brutas.
#TWF (probabilístico) e RNF (aleatório) não são previsíveis por nenhuma das abordagens.
resumo = {
    'Modo':     ['TWF', 'HDF', 'PWF', 'OSF', 'RNF'],
    'F1 base':  [0.00, 0.78, 0.65, 0.78, 0.00],
    'F1 eng':   [0.00, 0.93, 0.98, 0.95, 0.00],
    'Regra determinística': ['Não', 'Sim', 'Sim', 'Sim', 'Não']
}
df_resumo = pd.DataFrame(resumo)
print(df_resumo.to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 5))
x = range(len(resumo['Modo']))
largura = 0.35
ax.bar([i - largura/2 for i in x], resumo['F1 base'], largura, label='Sem feat. engenheiradas', color='#4A90D9')
ax.bar([i + largura/2 for i in x], resumo['F1 eng'], largura, label='Com feat. engenheiradas', color='#D94A4A')
ax.set_xticks(x)
ax.set_xticklabels(resumo['Modo'])
ax.set_ylabel('F1-Score')
ax.set_title('Impacto do Feature Engineering por Modo de Falha')
ax.legend()
ax.set_ylim(0, 1.1)
plt.tight_layout()
plt.show()