/* Manutenção Preditiva (AI4I 2020): versão estática do dashboard. Lê data/ai4i.json (gerado
   pelo exportar.py) e desenha tudo no navegador. Pontos em canvas (são 10.000), eixos em SVG. */
(function () {
  "use strict";

  const NOMES = { TWF: "Desgaste da ferramenta", HDF: "Dissipação de calor", PWF: "Potência", OSF: "Sobrecarga", RNF: "Aleatória" };
  const FORMULAS = {
    HDF: "ΔT < 8,6 K e rotação < 1.380 rpm",
    PWF: "potência < 3.500 W ou > 9.000 W",
    OSF: "desgaste × torque > limiar (L 11.000 · M 12.000 · H 13.000)",
    TWF: "desgaste entre 200 e 240 min",
  };
  const ROTULO_FEATURE = {
    "Air temperature [K]": "Temp. do ar",
    "Process temperature [K]": "Temp. do processo",
    "Rotational speed [rpm]": "Rotação",
    "Torque [Nm]": "Torque",
    "Tool wear [min]": "Desgaste",
    Type_encoded: "Tipo de produto",
    delta_temp: "ΔT ★",
    power_W: "Potência ★",
    wear_torque: "Desgaste × torque ★",
  };
  const DERIVADAS = new Set(["delta_temp", "power_W", "wear_torque"]);

  const num = (casas) => new Intl.NumberFormat("pt-BR", { minimumFractionDigits: casas, maximumFractionDigits: casas });
  const pct1 = new Intl.NumberFormat("pt-BR", { style: "percent", minimumFractionDigits: 1, maximumFractionDigits: 1 });
  const inteiro = new Intl.NumberFormat("pt-BR");
  const $ = (id) => document.getElementById(id);
  const cor = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();

  let D = null;
  let linhas = [];
  const estado = { tipos: new Set(["L", "M", "H"]), regra: "HDF", modo: "PWF" };

  /* ---------- utilidades ---------- */

  const tip = $("tooltip");
  function mostrarTip(ev, titulo, itens) {
    tip.replaceChildren();
    const t = document.createElement("div");
    t.className = "t-titulo";
    t.textContent = titulo;
    tip.appendChild(t);
    for (const it of itens) {
      const l = document.createElement("div");
      l.className = "t-linha";
      if (it.cor) {
        const i = document.createElement("i");
        i.style.background = it.cor;
        l.appendChild(i);
      }
      const b = document.createElement("b");
      b.textContent = it.valor;
      const s = document.createElement("span");
      s.textContent = it.rotulo;
      l.append(b, s);
      tip.appendChild(l);
    }
    tip.style.opacity = 1;
    tip.style.left = `${Math.min(ev.pageX + 14, window.scrollX + document.documentElement.clientWidth - tip.offsetWidth - 8)}px`;
    tip.style.top = `${ev.pageY + 14}px`;
  }
  const esconderTip = () => (tip.style.opacity = 0);

  function tabela(container, cab, corpo) {
    const t = document.createElement("table");
    const h = t.createTHead().insertRow();
    cab.forEach((c, i) => {
      const th = document.createElement("th");
      th.textContent = c;
      if (i > 0) th.className = "num";
      h.appendChild(th);
    });
    const b = t.createTBody();
    for (const l of corpo) {
      const tr = b.insertRow();
      l.forEach((c, i) => {
        const td = tr.insertCell();
        td.textContent = c;
        if (i > 0) td.className = "num";
      });
    }
    container.replaceChildren(t);
  }

  function legenda(container, itens) {
    container.replaceChildren(
      ...itens.map((it) => {
        const s = document.createElement("span");
        const i = document.createElement("i");
        if (it.bloco) i.className = "bloco";
        if (it.ponto) Object.assign(i.style, { width: "9px", height: "9px", borderRadius: "50%" });
        i.style.background = it.cor;
        s.append(i, document.createTextNode(it.rotulo));
        return s;
      })
    );
  }

  const larguraDe = (el) => Math.max(300, Math.floor(el.clientWidth));
  function svgEm(el, w, h, aria) {
    el.replaceChildren();
    return d3.select(el).append("svg").attr("viewBox", `0 0 ${w} ${h}`).attr("role", "img").attr("aria-label", aria);
  }
  const temModo = (i, modo) => (linhas[i].modos >> D.modos.indexOf(modo)) & 1;

  /** Barras verticais com base reta e ponta arredondada. dados: [{rotulo, valor, texto}] */
  function colunas(el, dados, { formato, aria, corBarra = "var(--series-1)" }) {
    const w = larguraDe(el);
    const h = 240;
    const m = { t: 24, r: 8, b: 28, l: 8 };
    const svg = svgEm(el, w, h, aria);
    const x = d3.scaleBand().domain(dados.map((d) => d.rotulo)).range([m.l, w - m.r]).padding(0.35);
    const y = d3.scaleLinear().domain([0, d3.max(dados, (d) => d.valor) * 1.15 || 1]).range([h - m.b, m.t]);
    const largura = Math.min(48, x.bandwidth());
    svg.append("line").attr("x1", m.l).attr("x2", w - m.r).attr("y1", y(0)).attr("y2", y(0)).style("stroke", "var(--axis)");
    const g = svg.selectAll("g.col").data(dados).join("g").attr("transform", (d) => `translate(${x(d.rotulo) + (x.bandwidth() - largura) / 2},0)`);
    g.append("path")
      .attr("d", (d) => {
        const topo = y(d.valor);
        const alt = y(0) - topo;
        const r = Math.min(4, alt, largura / 2);
        return `M0,${y(0)}v${-(alt - r)}a${r},${r} 0 0 1 ${r},${-r}h${largura - 2 * r}a${r},${r} 0 0 1 ${r},${r}v${alt - r}z`;
      })
      .style("fill", corBarra);
    g.append("text").attr("x", largura / 2).attr("y", (d) => y(d.valor) - 6).attr("text-anchor", "middle").attr("class", "rotulo-forte").text((d) => formato(d.valor));
    g.append("text").attr("x", largura / 2).attr("y", h - 8).attr("text-anchor", "middle").text((d) => d.rotulo);
    g.append("rect")
      .attr("x", -8)
      .attr("y", m.t)
      .attr("width", largura + 16)
      .attr("height", h - m.t - m.b)
      .style("fill", "transparent")
      .on("pointermove", (ev, d) => mostrarTip(ev, d.titulo || d.rotulo, [{ valor: formato(d.valor), rotulo: d.detalhe || "" }]))
      .on("pointerleave", esconderTip);
  }

  /** Dispersão: pontos em canvas, eixos e linhas de referência em SVG por cima. */
  function dispersao(el, { idx, x, y, xRotulo, yRotulo, destaque, referencias = [], aria, tituloTip }) {
    const w = larguraDe(el);
    const h = Math.max(280, Math.round(w * 0.5));
    const m = { t: 12, r: 16, b: 40, l: 56 };
    el.replaceChildren();
    el.style.position = "relative";
    const dpr = window.devicePixelRatio || 1;
    const canvas = document.createElement("canvas");
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    canvas.style.aspectRatio = `${w} / ${h}`;
    canvas.setAttribute("aria-hidden", "true");
    el.appendChild(canvas);
    const sx = d3.scaleLinear().domain(d3.extent(idx, (i) => x(i))).nice().range([m.l, w - m.r]);
    const sy = d3.scaleLinear().domain(d3.extent(idx, (i) => y(i))).nice().range([h - m.b, m.t]);

    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);
    const normais = idx.filter((i) => !destaque(i));
    const marcados = idx.filter((i) => destaque(i));
    ctx.fillStyle = cor("--neutral");
    ctx.globalAlpha = 0.45;
    for (const i of normais) ctx.fillRect(sx(x(i)) - 1.25, sy(y(i)) - 1.25, 2.5, 2.5);
    // Destaques por cima, com anel na cor da superfície para se separarem da nuvem
    ctx.globalAlpha = 1;
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = cor("--surface");
    ctx.fillStyle = cor("--series-2");
    for (const i of marcados) {
      ctx.beginPath();
      ctx.arc(sx(x(i)), sy(y(i)), 3.5, 0, 2 * Math.PI);
      ctx.fill();
      ctx.stroke();
    }

    const svg = d3.select(el).append("svg").attr("viewBox", `0 0 ${w} ${h}`).attr("role", "img").attr("aria-label", aria).style("position", "absolute").style("inset", 0);
    svg.append("g").attr("class", "eixo").attr("transform", `translate(0,${h - m.b})`).call(d3.axisBottom(sx).ticks(w < 600 ? 5 : 8).tickFormat((v) => inteiro.format(v)).tickSize(0).tickPadding(8));
    svg.append("g").attr("class", "eixo").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(sy).ticks(6).tickFormat((v) => inteiro.format(v)).tickSize(0).tickPadding(8));
    svg.append("text").attr("x", (m.l + w - m.r) / 2).attr("y", h - 4).attr("text-anchor", "middle").text(xRotulo);
    svg.append("text").attr("transform", `translate(14,${(m.t + h - m.b) / 2}) rotate(-90)`).attr("text-anchor", "middle").text(yRotulo);

    for (const ref of referencias) {
      const [x0, x1] = sx.domain();
      const [y0, y1] = sy.domain();
      // Reta (2 pontos): corta nas bordas do gráfico. Curva: descarta o trecho fora, para não "escorrer" pela borda.
      const pontos =
        ref.pontos.length === 2
          ? ref.pontos.map(([a, b]) => [Math.min(x1, Math.max(x0, a)), Math.min(y1, Math.max(y0, b))])
          : ref.pontos.filter(([a, b]) => b >= y0 && b <= y1 && a >= x0 && a <= x1);
      if (pontos.length < 2) continue;
      svg.append("path").attr("d", d3.line()(pontos.map(([a, b]) => [sx(a), sy(b)]))).style("fill", "none").style("stroke", "var(--ink)").style("stroke-width", 1.5);
      const [ax, ay] = pontos[ref.rotuloEm === "inicio" ? 0 : pontos.length - 1];
      const perto_da_direita = sx(ax) > w - m.r - 90;
      svg
        .append("text")
        .attr("x", sx(ax) + (perto_da_direita ? -4 : 4))
        .attr("y", Math.max(m.t + 10, sy(ay) - 6))
        .attr("text-anchor", perto_da_direita ? "end" : "start")
        .attr("class", "rotulo-forte")
        .text(ref.rotulo);
    }

    // Dica pelo ponto mais próximo do ponteiro (Delaunay): ninguém precisa acertar um ponto de 3 px
    const delaunay = d3.Delaunay.from(idx, (i) => sx(x(i)), (i) => sy(y(i)));
    svg
      .append("rect")
      .attr("x", m.l)
      .attr("y", m.t)
      .attr("width", w - m.l - m.r)
      .attr("height", h - m.t - m.b)
      .style("fill", "transparent")
      .on("pointermove", (ev) => {
        const [px, py] = d3.pointer(ev);
        const i = idx[delaunay.find(px, py)];
        const d = Math.hypot(sx(x(i)) - px, sy(y(i)) - py);
        if (d > 24) return esconderTip();
        mostrarTip(ev, tituloTip(i), [
          { valor: inteiro.format(x(i)), rotulo: xRotulo },
          { valor: num(1).format(y(i)), rotulo: yRotulo },
        ]);
      })
      .on("pointerleave", esconderTip);
  }

  /* ---------- seções ---------- */

  function filtrados() {
    return d3.range(linhas.length).filter((i) => estado.tipos.has(linhas[i].tipo));
  }

  function renderVisao() {
    const idx = filtrados();
    const falhas = idx.filter((i) => linhas[i].falha).length;
    const porModo = D.modos.map((m) => ({ modo: m, n: idx.filter((i) => temModo(i, m)).length }));
    const topo = porModo.reduce((a, b) => (b.n > a.n ? b : a), porModo[0]);
    const tiles = [
      { rotulo: "Registros", valor: inteiro.format(idx.length) },
      { rotulo: "Falhas (parada da máquina)", valor: inteiro.format(falhas) },
      { rotulo: "Taxa de falha", valor: idx.length ? pct1.format(falhas / idx.length) : "—" },
      { rotulo: "Modo mais frequente", valor: idx.length ? topo.modo : "—", detalhe: idx.length ? `${NOMES[topo.modo]} · ${topo.n} ocorrências` : "" },
    ];
    $("tiles").replaceChildren(...tiles.map(tile));

    const tipos = ["L", "M", "H"].filter((t) => estado.tipos.has(t)).map((t) => {
      const it = idx.filter((i) => linhas[i].tipo === t);
      const f = it.filter((i) => linhas[i].falha).length;
      return { rotulo: t, valor: it.length ? f / it.length : 0, titulo: `Tipo ${t}`, detalhe: `${f} falhas em ${inteiro.format(it.length)} registros` };
    });
    colunas($("g-tipo"), tipos, { formato: (v) => pct1.format(v), aria: "Taxa de falha por tipo de produto" });
    colunas(
      $("g-modos"),
      porModo.map((p) => ({ rotulo: p.modo, valor: p.n, titulo: `${p.modo} · ${NOMES[p.modo]}`, detalhe: "ocorrências" })),
      { formato: (v) => inteiro.format(v), aria: "Ocorrências por modo de falha" }
    );

    legenda($("leg-dispersao"), [
      { cor: "var(--neutral)", ponto: true, rotulo: "Sem falha" },
      { cor: "var(--series-2)", ponto: true, rotulo: "Com falha" },
    ]);
    dispersao($("g-dispersao"), {
      idx,
      x: (i) => linhas[i].rpm,
      y: (i) => linhas[i].torque,
      xRotulo: "Rotação (rpm)",
      yRotulo: "Torque (Nm)",
      destaque: (i) => linhas[i].falha,
      aria: "Dispersão de torque por rotação, com as falhas destacadas",
      tituloTip: (i) => `Tipo ${linhas[i].tipo} · ${linhas[i].falha ? "com falha" : "sem falha"}`,
    });
    tabela(
      $("t-visao"),
      ["Tipo", "Registros", "Falhas", "Taxa de falha"],
      tipos.map((t) => {
        const it = idx.filter((i) => linhas[i].tipo === t.rotulo);
        return [t.rotulo, inteiro.format(it.length), it.filter((i) => linhas[i].falha).length, pct1.format(t.valor)];
      })
    );
  }

  function tile(t) {
    const c = document.createElement("div");
    c.className = "cartao tile";
    const a = document.createElement("div");
    a.className = "rotulo";
    a.textContent = t.rotulo;
    const v = document.createElement("div");
    v.className = `valor ${t.classe || ""}`;
    v.textContent = t.valor;
    c.append(a, v);
    if (t.detalhe) {
      const d = document.createElement("div");
      d.className = "detalhe";
      d.textContent = t.detalhe;
      c.appendChild(d);
    }
    return c;
  }

  function renderTabelaRegras() {
    const t = document.createElement("table");
    t.className = "tabela-regras";
    const h = t.createTHead().insertRow();
    ["Modo", "Regra reconstruída", "Concordância", "Falsos +", "Falsos −", "Tipo"].forEach((c, i) => {
      const th = document.createElement("th");
      th.textContent = c;
      if (i >= 2 && i <= 4) th.className = "num";
      h.appendChild(th);
    });
    const b = t.createTBody();
    for (const r of D.regras) {
      const tr = b.insertRow();
      const c0 = tr.insertCell();
      const forte = document.createElement("strong");
      forte.textContent = r.modo;
      c0.append(forte, document.createTextNode(` ${NOMES[r.modo]}`));
      const c1 = tr.insertCell();
      const code = document.createElement("code");
      code.textContent = FORMULAS[r.modo];
      c1.appendChild(code);
      [pct1.format(r.concordancia), inteiro.format(r.falsos_positivos), inteiro.format(r.falsos_negativos)].forEach((v) => {
        const td = tr.insertCell();
        td.className = "num";
        td.textContent = v;
      });
      const c5 = tr.insertCell();
      const selo = document.createElement("span");
      selo.className = "selo-regra";
      selo.textContent = r.deterministica ? "Determinística" : "Probabilística";
      c5.appendChild(selo);
    }
    $("t-regras").replaceChildren(t);
  }

  function renderRegra() {
    const regra = estado.regra;
    const idx = d3.range(linhas.length);
    const rpmFaixa = d3.range(1100, 3001, 20);
    const config = {
      HDF: {
        titulo: "HDF: ΔT × rotação",
        sub: "Falha quando a diferença entre a temperatura do processo e a do ar fica abaixo de 8,6 K com rotação abaixo de 1.380 rpm.",
        x: (i) => linhas[i].rpm,
        y: (i) => linhas[i].delta_temp,
        xRotulo: "Rotação (rpm)",
        yRotulo: "ΔT (K)",
        referencias: [
          { pontos: [[1000, 8.6], [3000, 8.6]], rotulo: "ΔT = 8,6 K" },
          { pontos: [[1380, 0], [1380, 14]], rotulo: "1.380 rpm", rotuloEm: "fim" },
        ],
      },
      PWF: {
        titulo: "PWF: torque × rotação",
        sub: "Falha quando a potência (torque × velocidade angular) sai da faixa de 3.500 a 9.000 W. As curvas são essas potências constantes.",
        x: (i) => linhas[i].rpm,
        y: (i) => linhas[i].torque,
        xRotulo: "Rotação (rpm)",
        yRotulo: "Torque (Nm)",
        referencias: [3500, 9000].map((p) => ({ pontos: rpmFaixa.map((r) => [r, p / ((r * 2 * Math.PI) / 60)]), rotulo: `${inteiro.format(p)} W`, rotuloEm: "fim" })),
      },
      OSF: {
        titulo: "OSF: torque × desgaste",
        sub: "Falha quando desgaste × torque passa do limiar do tipo de produto. As curvas são os três limiares.",
        x: (i) => linhas[i].desgaste,
        y: (i) => linhas[i].torque,
        xRotulo: "Desgaste da ferramenta (min)",
        yRotulo: "Torque (Nm)",
        referencias: Object.entries(D.limiar_osf).map(([tipo, lim]) => ({
          pontos: d3.range(120, 260, 2).map((wv) => [wv, lim / wv]),
          rotulo: `${tipo} ${inteiro.format(lim)}`,
          rotuloEm: "fim",
        })),
      },
    }[regra];
    $("titulo-regra").textContent = config.titulo;
    $("sub-regra").textContent = config.sub;
    legenda($("leg-regra"), [
      { cor: "var(--neutral)", ponto: true, rotulo: `Sem ${regra}` },
      { cor: "var(--series-2)", ponto: true, rotulo: `Com ${regra} (rótulo real do dataset)` },
      { cor: "var(--ink)", rotulo: "Limite da regra documentada" },
    ]);
    dispersao($("g-regra"), {
      idx,
      ...config,
      destaque: (i) => temModo(i, regra),
      aria: config.titulo,
      tituloTip: (i) => `Tipo ${linhas[i].tipo} · ${temModo(i, regra) ? `com ${regra}` : `sem ${regra}`}`,
    });
  }

  function renderClassificacao() {
    const { f1_base: base, f1_eng: eng, importancia_eng: imp } = D.modelos;
    $("tiles-f1").replaceChildren(
      ...D.modos.map((m) => {
        const delta = eng[m] - base[m];
        return tile({
          rotulo: `${m} · ${NOMES[m]}`,
          valor: num(2).format(eng[m]),
          classe: delta > 0 ? "desce" : "",
          detalhe: delta > 0 ? `+${num(2).format(delta)} vs. sem as features (${num(2).format(base[m])})` : `igual sem as features (${num(2).format(base[m])})`,
        });
      })
    );

    // F1 agrupado: cinza = só variáveis brutas, azul = com as features derivadas da física
    const el = $("g-f1");
    legenda($("leg-f1"), [
      { cor: "var(--neutral)", bloco: true, rotulo: "Só variáveis brutas" },
      { cor: "var(--series-1)", bloco: true, rotulo: "Com features derivadas" },
    ]);
    const w = larguraDe(el);
    const h = 260;
    const m = { t: 22, r: 8, b: 28, l: 36 };
    const svg = svgEm(el, w, h, "F1 por modo de falha, com e sem as features derivadas");
    const x0 = d3.scaleBand().domain(D.modos).range([m.l, w - m.r]).padding(0.3);
    const x1 = d3.scaleBand().domain(["base", "eng"]).range([0, x0.bandwidth()]).padding(0.08);
    const y = d3.scaleLinear().domain([0, 1]).range([h - m.b, m.t]);
    svg.append("g").attr("class", "grade").selectAll("line").data(y.ticks(5)).join("line").attr("x1", m.l).attr("x2", w - m.r).attr("y1", y).attr("y2", y);
    svg.append("g").attr("class", "eixo").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(5).tickFormat((v) => num(1).format(v)).tickSize(0).tickPadding(6));
    const grupos = svg.selectAll("g.grupo").data(D.modos).join("g").attr("transform", (d) => `translate(${x0(d)},0)`);
    grupos.append("text").attr("x", x0.bandwidth() / 2).attr("y", h - 8).attr("text-anchor", "middle").text((d) => d);
    const barras = grupos
      .selectAll("g.barra")
      .data((modo) => ["base", "eng"].map((k) => ({ modo, k, v: k === "base" ? base[modo] : eng[modo] })))
      .join("g");
    const larg = Math.min(24, x1.bandwidth());
    barras
      .append("path")
      .attr("d", (d) => {
        const x = x1(d.k) + (x1.bandwidth() - larg) / 2;
        const alt = y(0) - y(d.v);
        if (alt <= 0) return "";
        const r = Math.min(4, alt, larg / 2);
        return `M${x},${y(0)}v${-(alt - r)}a${r},${r} 0 0 1 ${r},${-r}h${larg - 2 * r}a${r},${r} 0 0 1 ${r},${r}v${alt - r}z`;
      })
      .style("fill", (d) => (d.k === "eng" ? "var(--series-1)" : "var(--neutral)"));
    barras
      .filter((d) => d.k === "eng")
      .append("text")
      .attr("x", (d) => x1(d.k) + x1.bandwidth() / 2)
      .attr("y", (d) => y(d.v) - 6)
      .attr("text-anchor", "middle")
      .attr("class", "rotulo-forte")
      .text((d) => num(2).format(d.v));
    grupos
      .append("rect")
      .attr("width", x0.bandwidth())
      .attr("y", m.t)
      .attr("height", h - m.t - m.b)
      .style("fill", "transparent")
      .on("pointermove", (ev, modo) =>
        mostrarTip(ev, `${modo} · ${NOMES[modo]}`, [
          { cor: "var(--series-1)", valor: num(2).format(eng[modo]), rotulo: "com features derivadas" },
          { cor: "var(--neutral)", valor: num(2).format(base[modo]), rotulo: "só variáveis brutas" },
        ])
      )
      .on("pointerleave", esconderTip);

    renderImportancia();
    tabela(
      $("t-f1"),
      ["Modo", "F1 só brutas", "F1 com derivadas", "Diferença"],
      D.modos.map((m) => [`${m} · ${NOMES[m]}`, num(2).format(base[m]), num(2).format(eng[m]), num(2).format(eng[m] - base[m])])
    );
  }

  function renderImportancia() {
    const imp = D.modelos.importancia_eng[estado.modo];
    const dados = Object.entries(imp)
      .map(([f, v]) => ({ f, v }))
      .sort((a, b) => b.v - a.v);
    const el = $("g-importancia");
    const w = larguraDe(el);
    const linha = 24;
    const m = { t: 4, r: 52, b: 8, l: 130 };
    const h = m.t + m.b + dados.length * linha;
    const svg = svgEm(el, w, h, `Importância das features no modo ${estado.modo}`);
    const x = d3.scaleLinear().domain([0, d3.max(dados, (d) => d.v) || 1]).range([m.l, w - m.r]);
    const y = d3.scaleBand().domain(dados.map((d) => d.f)).range([m.t, h - m.b]).padding(0.3);
    const g = svg.selectAll("g").data(dados).join("g").attr("transform", (d) => `translate(0,${y(d.f)})`);
    g.append("text").attr("x", m.l - 8).attr("y", y.bandwidth() / 2).attr("dy", "0.35em").attr("text-anchor", "end").attr("class", (d) => (DERIVADAS.has(d.f) ? "rotulo-forte" : null)).text((d) => ROTULO_FEATURE[d.f] || d.f);
    g.append("path")
      .attr("d", (d) => {
        const wv = Math.max(0, x(d.v) - x(0));
        const r = Math.min(4, wv, y.bandwidth() / 2);
        return `M${x(0)},0h${wv - r}a${r},${r} 0 0 1 ${r},${r}v${y.bandwidth() - 2 * r}a${r},${r} 0 0 1 -${r},${r}h-${wv - r}z`;
      })
      .style("fill", (d) => (DERIVADAS.has(d.f) ? "var(--series-1)" : "var(--neutral)"));
    g.append("text").attr("x", (d) => x(d.v) + 6).attr("y", y.bandwidth() / 2).attr("dy", "0.35em").text((d) => num(3).format(d.v));
  }

  /* ---------- controles ---------- */

  function montarControles() {
    const grupo = $("f-tipos");
    for (const t of ["L", "M", "H"]) {
      const b = document.createElement("button");
      b.type = "button";
      b.textContent = t;
      b.setAttribute("aria-pressed", "true");
      b.addEventListener("click", () => {
        if (estado.tipos.has(t) && estado.tipos.size === 1) return; // pelo menos um tipo
        estado.tipos.has(t) ? estado.tipos.delete(t) : estado.tipos.add(t);
        b.setAttribute("aria-pressed", String(estado.tipos.has(t)));
        renderVisao();
      });
      grupo.appendChild(b);
    }
    $("f-regra").addEventListener("change", (e) => {
      estado.regra = e.target.value;
      renderRegra();
    });
    const sel = $("f-modo");
    for (const m of D.modos) {
      const o = document.createElement("option");
      o.value = m;
      o.textContent = `${m} · ${NOMES[m]}`;
      o.selected = m === estado.modo;
      sel.appendChild(o);
    }
    sel.addEventListener("change", (e) => {
      estado.modo = e.target.value;
      renderImportancia();
    });

    const btn = $("btn-tema");
    const atual = () => document.documentElement.getAttribute("data-theme") || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
    const desenhar = () => {
      const escuro = atual() === "dark";
      btn.textContent = escuro ? "☀" : "☾";
      btn.setAttribute("aria-label", escuro ? "Mudar para o modo claro" : "Mudar para o modo escuro");
    };
    btn.addEventListener("click", () => {
      const novo = atual() === "dark" ? "light" : "dark";
      document.documentElement.setAttribute("data-theme", novo);
      try {
        localStorage.setItem("tema", novo);
      } catch (e) {
        /* sem storage, vale só nesta visita */
      }
      desenhar();
      renderTudo(); // o canvas lê as cores na hora de desenhar
    });
    matchMedia("(prefers-color-scheme: dark)").addEventListener("change", renderTudo);
    desenhar();
    let espera;
    window.addEventListener("resize", () => {
      clearTimeout(espera);
      espera = setTimeout(renderTudo, 150);
    });
  }

  function renderTudo() {
    renderVisao();
    renderRegra();
    renderClassificacao();
  }

  fetch("data/ai4i.json")
    .then((r) => r.json())
    .then((dados) => {
      D = dados;
      const r = dados.registros;
      linhas = r.tipo.map((tipo, i) => ({ tipo, rpm: r.rpm[i], torque: r.torque[i], desgaste: r.desgaste[i], delta_temp: r.delta_temp[i], falha: r.falha[i], modos: r.modos[i] }));
      montarControles();
      renderTabelaRegras();
      renderTudo();
    })
    .catch(() => {
      document.querySelector(".abertura .selo").textContent = "Não foi possível carregar os dados. Tente recarregar a página.";
    });
})();
