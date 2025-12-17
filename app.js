// =======================
// Ajuste automático en JS
// =======================

// ---- Métricas ----
function mean(arr){ return arr.reduce((a,b)=>a+b,0)/arr.length; }
function mse(y, yhat){ let s=0; for(let i=0;i<y.length;i++){ const e=y[i]-yhat[i]; s+=e*e; } return s/y.length; }
function rmse(y,yhat){ return Math.sqrt(mse(y,yhat)); }
function mae(y,yhat){ let s=0; for(let i=0;i<y.length;i++){ s+=Math.abs(y[i]-yhat[i]); } return s/y.length; }
function r2(y,yhat){
  const ybar=mean(y); let ssRes=0, ssTot=0;
  for(let i=0;i<y.length;i++){ const e=y[i]-yhat[i]; ssRes+=e*e; const d=y[i]-ybar; ssTot+=d*d; }
  return 1 - (ssRes/ssTot);
}
function bic(y,yhat,k){
  const n=y.length; let rss=0;
  for(let i=0;i<n;i++){ const e=y[i]-yhat[i]; rss+=e*e; }
  rss=Math.max(rss,1e-12);
  return n*Math.log(rss/n) + k*Math.log(n);
}

// ---- Split reproducible ----
function mulberry32(a){ return function(){ let t=a+=0x6D2B79F5; t=Math.imul(t^t>>>15,t|1); t^=t+Math.imul(t^t>>>7,t|61); return ((t^t>>>14)>>>0)/4294967296; } }
function trainTestSplit(t,x,testRatio=0.25,seed=42){
  const n=t.length; const idx=[...Array(n).keys()];
  const rnd=mulberry32(seed);
  for(let i=n-1;i>0;i--){ const j=Math.floor(rnd()*(i+1)); [idx[i],idx[j]]=[idx[j],idx[i]]; }
  const nTest=Math.max(1, Math.floor(n*testRatio));
  const testIdx=idx.slice(0,nTest), trainIdx=idx.slice(nTest);
  const pick=(arr,ids)=>ids.map(i=>arr[i]);
  return { tTrain:pick(t,trainIdx), xTrain:pick(x,trainIdx), tTest:pick(t,testIdx), xTest:pick(x,testIdx) };
}

// ---- Resolver sistema lineal (Gauss) para polinomios ----
function solveLinearSystem(A,b){
  const n=A.length;
  const M=A.map((row,i)=>row.concat([b[i]]));
  for(let col=0; col<n; col++){
    let pivot=col;
    for(let r=col+1;r<n;r++) if(Math.abs(M[r][col])>Math.abs(M[pivot][col])) pivot=r;
    [M[col],M[pivot]]=[M[pivot],M[col]];
    const diag=M[col][col] || 1e-12;
    for(let c=col;c<=n;c++) M[col][c]/=diag;
    for(let r=0;r<n;r++){
      if(r===col) continue;
      const factor=M[r][col];
      for(let c=col;c<=n;c++) M[r][c]-=factor*M[col][c];
    }
  }
  return M.map(row=>row[n]);
}
function polyFit(t, x, degree){
  const m=degree+1;
  const XtX=[...Array(m)].map(()=>Array(m).fill(0));
  const Xty=Array(m).fill(0);
  for(let i=0;i<t.length;i++){
    const powers=Array(m).fill(1);
    for(let j=1;j<m;j++) powers[j]=powers[j-1]*t[i];
    for(let r=0;r<m;r++){
      Xty[r]+=powers[r]*x[i];
      for(let c=0;c<m;c++) XtX[r][c]+=powers[r]*powers[c];
    }
  }
  return solveLinearSystem(XtX,Xty); // [c0,c1,c2...]
}
function polyPredict(t, coeffs){
  return t.map(v=>{
    let s=0, p=1;
    for(let i=0;i<coeffs.length;i++){ s+=coeffs[i]*p; p*=v; }
    return s;
  });
}

// ---- Nelder–Mead (para saturación y exponencial) ----
function nelderMead(f, x0, opts={}){
  const maxIter=opts.maxIter ?? 1600;
  const step=opts.step ?? 0.25;
  const alpha=1, gamma=2, rho=0.5, sigma=0.5;
  const n=x0.length;

  let simplex=[x0.slice()];
  for(let i=0;i<n;i++){
    const x=x0.slice();
    x[i]=x[i]===0 ? step : x[i]*(1+step);
    simplex.push(x);
  }
  function sortSimplex(){ simplex.sort((a,b)=>f(a)-f(b)); }
  sortSimplex();

  for(let iter=0; iter<maxIter; iter++){
    sortSimplex();
    const best=simplex[0], worst=simplex[n], secondWorst=simplex[n-1];

    const centroid=Array(n).fill(0);
    for(let i=0;i<n;i++) for(let j=0;j<n;j++) centroid[j]+=simplex[i][j];
    for(let j=0;j<n;j++) centroid[j]/=n;

    const reflect=centroid.map((c,j)=>c+alpha*(c-worst[j]));
    const fBest=f(best), fSecond=f(secondWorst), fWorst=f(worst), fReflect=f(reflect);

    if(fReflect < fBest){
      const expand=centroid.map((c,j)=>c+gamma*(reflect[j]-c));
      simplex[n] = (f(expand) < fReflect) ? expand : reflect;
      continue;
    }
    if(fReflect < fSecond){ simplex[n]=reflect; continue; }

    const contract=centroid.map((c,j)=>c+rho*(worst[j]-c));
    if(f(contract) < fWorst){ simplex[n]=contract; continue; }

    for(let i=1;i<simplex.length;i++){
      simplex[i]=simplex[i].map((v,j)=>best[j]+sigma*(v-best[j]));
    }
  }
  sortSimplex();
  return simplex[0];
}
function sse(y,yhat){ let s=0; for(let i=0;i<y.length;i++){ const e=y[i]-yhat[i]; s+=e*e; } return s; }

// Saturación: x(t)=L - B exp(-k t), k>0 usando k=exp(u)
function fitSaturation(t,x){
  const L0=Math.max(...x);
  const B0=L0 - x[0];
  const u0=Math.log(0.5);
  const x0=[L0, B0, u0];

  const f=(p)=>{
    const L=p[0], B=p[1], k=Math.exp(p[2]);
    const pred=t.map(tt=> L - B*Math.exp(-k*tt));
    return sse(x,pred);
  };
  const p=nelderMead(f,x0,{maxIter:1800,step:0.25});
  return {L:p[0], B:p[1], k:Math.exp(p[2])};
}

// Exponencial: x(t)=A exp(k t)+C
function fitExponential(t,x){
  const A0=x[0] || 1, k0=0.1, C0=mean(x);
  const x0=[A0,k0,C0];

  const f=(p)=>{
    const A=p[0], k=p[1], C=p[2];
    const pred=t.map(tt=> A*Math.exp(k*tt)+C);
    for(const v of pred) if(!Number.isFinite(v)) return 1e30;
    return sse(x,pred);
  };
  const p=nelderMead(f,x0,{maxIter:1800,step:0.25});
  return {A:p[0], k:p[1], C:p[2]};
}

// ---- UI ----
const elFile=document.getElementById('file');
const elFileName=document.getElementById('fileName');
const elRun=document.getElementById('run');
const elDemo=document.getElementById('demo');
const elMode=document.getElementById('mode');
const elStatus=document.getElementById('status');
const elCols=document.getElementById('cols');
const elBestModel=document.getElementById('bestModel');
const elEq=document.getElementById('equation');
const elMetrics=document.getElementById('metrics');
const elTbody=document.getElementById('tbody');

let parsedData=null;
function setStatus(msg){ elStatus.textContent=msg; }
function fmt(v){ return (Math.round(v*1e6)/1e6).toString(); }

function chooseBetter(a,b,eps=1e-4){
  if(!b) return a;
  if(a.r2Test > b.r2Test + eps) return a;
  if(Math.abs(a.r2Test - b.r2Test) <= eps) return (a.bicTest < b.bicTest) ? a : b;
  return b;
}

function buildEquation(best){
  if(best.name==='Lineal'){
    const [c0,c1]=best.params; // x = c0 + c1 t
    return `x(t) = m t + c\n\nm = ${fmt(c1)}\nc = ${fmt(c0)}\n\nReemplazando:\nx(t) = ${fmt(c1)} t + ${fmt(c0)}`;
  }
  if(best.name==='Cuadrática'){
    const [c0,c1,c2]=best.params;
    return `x(t) = a t^2 + b t + c\n\na = ${fmt(c2)}\nb = ${fmt(c1)}\nc = ${fmt(c0)}\n\nReemplazando:\nx(t) = ${fmt(c2)} t^2 + ${fmt(c1)} t + ${fmt(c0)}`;
  }
  if(best.name==='Cúbica'){
    const [c0,c1,c2,c3]=best.params;
    return `x(t) = a t^3 + b t^2 + c t + d\n\na = ${fmt(c3)}\nb = ${fmt(c2)}\nc = ${fmt(c1)}\nd = ${fmt(c0)}\n\nReemplazando:\nx(t) = ${fmt(c3)} t^3 + ${fmt(c2)} t^2 + ${fmt(c1)} t + ${fmt(c0)}`;
  }
  if(best.name==='Saturación (1er Orden)'){
    const {L,B,k}=best.paramsObj;
    return `x(t) = L - B e^(-k t)\n\nL = ${fmt(L)}\nB = ${fmt(B)}\nk = ${fmt(k)}\n\nReemplazando:\nx(t) = ${fmt(L)} - ${fmt(B)} e^(-${fmt(k)} t)`;
  }
  const {A,k,C}=best.paramsObj;
  return `x(t) = A e^(k t) + C\n\nA = ${fmt(A)}\nk = ${fmt(k)}\nC = ${fmt(C)}\n\nReemplazando:\nx(t) = ${fmt(A)} e^(${fmt(k)} t) + ${fmt(C)}`;
}

function fillTable(rows){
  elTbody.innerHTML='';
  for(const r of rows){
    const tr=document.createElement('tr');
    tr.innerHTML = `
      <td>${r.name}</td>
      <td>${r.r2Test.toFixed(4)}</td>
      <td>${r.rmseTest.toFixed(4)}</td>
      <td>${r.maeTest.toFixed(4)}</td>
      <td>${r.bicTest.toFixed(2)}</td>
    `;
    elTbody.appendChild(tr);
  }
}

function addMetricChips(best){
  elMetrics.innerHTML='';
  const chips=[
    ['R²', best.r2Test],
    ['RMSE', best.rmseTest],
    ['MAE', best.maeTest],
    ['BIC', best.bicTest]
  ];
  for(const [k,v] of chips){
    const s=document.createElement('span');
    s.textContent=`${k}: ${v.toFixed(4)}`;
    elMetrics.appendChild(s);
  }
}

function plotAll(t,x,tLine,yLine,bestLabel){
  const data=[
    {x:t,y:x,mode:'markers',name:'Datos reales',marker:{size:6}},
    {x:tLine,y:yLine,mode:'lines',name:`Mejor ajuste: ${bestLabel}`,line:{width:3}}
  ];
  const layout={
    title:'Ajuste de Curva Automático (web)',
    xaxis:{title:'t'},
    yaxis:{title:'x'},
    margin:{l:55,r:20,t:55,b:50},
    paper_bgcolor:'rgba(0,0,0,0)',
    plot_bgcolor:'rgba(0,0,0,0)',
    font:{color:'#e8eefc'}
  };
  Plotly.newPlot('plot',data,layout,{responsive:true,displaylogo:false});
}

function parseCsvFile(file){
  setStatus('Leyendo CSV...');
  Papa.parse(file,{
    header:true,
    dynamicTyping:true,
    skipEmptyLines:true,
    complete:(res)=>{
      const cols = res.meta.fields ?? Object.keys(res.data[0] ?? {});
      if(!cols || cols.length<2){ setStatus('Necesito al menos 2 columnas.'); return; }
      const c1=cols[0], c2=cols[1];

      let t=[], x=[];
      for(const row of res.data){
        const tt=Number(row[c1]), xx=Number(row[c2]);
        if(Number.isFinite(tt) && Number.isFinite(xx)){ t.push(tt); x.push(xx); }
      }
      if(t.length<8){ setStatus('Muy pocos datos numéricos (mínimo ~8).'); return; }

      // ordenar por t
      const idx=[...Array(t.length).keys()].sort((i,j)=>t[i]-t[j]);
      t=idx.map(i=>t[i]); x=idx.map(i=>x[i]);

      parsedData={t,x,colT:c1,colX:c2};
      elCols.textContent=`t='${c1}', x='${c2}' | filas=${t.length}`;
      elRun.disabled=false;
      setStatus('Listo. Pulsa “Calcular ajuste”.');
    },
    error:()=>setStatus('Error al parsear CSV.')
  });
}

elFile.addEventListener('change',(e)=>{
  const f=e.target.files?.[0];
  if(!f) return;
  elFileName.textContent=f.name;
  parseCsvFile(f);
});

// Demo
elDemo.addEventListener('click',()=>{
  const t=[...Array(101).keys()].map(i=>i/10);
  const L=10, B=9.9, k=0.5;
  const rnd=mulberry32(7);
  const x=t.map(tt=> (L - B*Math.exp(-k*tt)) + (rnd()-0.5)*0.4);
  parsedData={t,x,colT:'t',colX:'x'};
  elCols.textContent=`t='t', x='x' | filas=${t.length} (demo)`;
  elRun.disabled=false;
  setStatus('Demo cargada. Pulsa “Calcular ajuste”.');
});

elRun.addEventListener('click',()=>{
  if(!parsedData) return;
  const {t,x}=parsedData;
  const mode=elMode.value;

  setStatus('Ajustando modelos...');
  const split = (mode==='traintest') ? trainTestSplit(t,x,0.25,42) : {tTrain:t,xTrain:x,tTest:t,xTest:x};

  // 5 modelos
  const rows=[];

  // Lineal
  {
    const coeffs = polyFit(split.tTrain, split.xTrain, 1);
    const predTest = polyPredict(split.tTest, coeffs);
    rows.push({
      name:'Lineal',
      params: coeffs,
      paramsObj:null,
      r2Test: r2(split.xTest, predTest),
      rmseTest: rmse(split.xTest, predTest),
      maeTest: mae(split.xTest, predTest),
      bicTest: bic(split.xTest, predTest, 2),
      predictAll: (tt)=>polyPredict(tt, coeffs)
    });
  }

  // Cuadrática
  {
    const coeffs = polyFit(split.tTrain, split.xTrain, 2);
    const predTest = polyPredict(split.tTest, coeffs);
    rows.push({
      name:'Cuadrática',
      params: coeffs,
      paramsObj:null,
      r2Test: r2(split.xTest, predTest),
      rmseTest: rmse(split.xTest, predTest),
      maeTest: mae(split.xTest, predTest),
      bicTest: bic(split.xTest, predTest, 3),
      predictAll: (tt)=>polyPredict(tt, coeffs)
    });
  }

  // Cúbica
  {
    const coeffs = polyFit(split.tTrain, split.xTrain, 3);
    const predTest = polyPredict(split.tTest, coeffs);
    rows.push({
      name:'Cúbica',
      params: coeffs,
      paramsObj:null,
      r2Test: r2(split.xTest, predTest),
      rmseTest: rmse(split.xTest, predTest),
      maeTest: mae(split.xTest, predTest),
      bicTest: bic(split.xTest, predTest, 4),
      predictAll: (tt)=>polyPredict(tt, coeffs)
    });
  }

  // Saturación (LBk)
  {
    const p = fitSaturation(split.tTrain, split.xTrain);
    const predTest = split.tTest.map(tt => p.L - p.B*Math.exp(-p.k*tt));
    rows.push({
      name:'Saturación (1er Orden)',
      params: null,
      paramsObj: p,
      r2Test: r2(split.xTest, predTest),
      rmseTest: rmse(split.xTest, predTest),
      maeTest: mae(split.xTest, predTest),
      bicTest: bic(split.xTest, predTest, 3),
      predictAll: (tt)=>tt.map(v => p.L - p.B*Math.exp(-p.k*v))
    });
  }

  // Exponencial
  {
    const p = fitExponential(split.tTrain, split.xTrain);
    const predTest = split.tTest.map(tt => p.A*Math.exp(p.k*tt) + p.C);
    rows.push({
      name:'Exponencial',
      params: null,
      paramsObj: p,
      r2Test: r2(split.xTest, predTest),
      rmseTest: rmse(split.xTest, predTest),
      maeTest: mae(split.xTest, predTest),
      bicTest: bic(split.xTest, predTest, 3),
      predictAll: (tt)=>tt.map(v => p.A*Math.exp(p.k*v) + p.C)
    });
  }

  // Elegir mejor: R², y si empata, BIC
  let best=null;
  for(const row of rows) best = chooseBetter(row, best);

  // UI
  elBestModel.textContent = best.name;
  elEq.textContent = buildEquation(best);
  addMetricChips(best);

  // tabla ordenada por R² desc
  fillTable(rows.slice().sort((a,b)=>b.r2Test-a.r2Test));

  // curva suave
  const tMin=Math.min(...t), tMax=Math.max(...t);
  const tLine=[...Array(500).keys()].map(i => tMin + (tMax-tMin)*i/499);
  const yLine = best.predictAll(tLine);

  plotAll(t,x,tLine,yLine,best.name);
  setStatus('Listo ✅');
});
