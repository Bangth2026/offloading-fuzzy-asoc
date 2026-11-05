const fs=require('fs');
const d=require('docx');
const {Document,Packer,Paragraph,TextRun,ImageRun,Table,TableRow,TableCell,WidthType,AlignmentType,HeadingLevel,ShadingType,BorderStyle,LevelFormat,PageBreak}=d;
const [src,out]=process.argv.slice(2);
const L=fs.readFileSync(src,'utf8').split('\n');
const F="Times New Roman";
function mathRuns(src,base){
  // $...$ : letters italic, _x/_{..} subscript, ^x/^{..} superscript
  const out=[]; let k=0;
  const push=(t,o)=>{ for(const piece of t.split(/([A-Za-zα-ωΑ-Ωμπρ]+)/)){ if(!piece) continue;
      const it=/^[A-Za-zα-ωμπρ]+$/.test(piece) && !/^(AUC|SC|SM|TSK|LOW|HIGH|MED|max|min|log|sign|clip|argmin|exp)$/.test(piece);
      out.push(new TextRun({text:piece,font:F,italics:it,...base,...o})); } };
  while(k<src.length){
    const c=src[k];
    if(c==='_'||c==='^'){ let t; if(src[k+1]==='{'){ const e=src.indexOf('}',k+2); t=src.slice(k+2,e); k=e+1;} else { t=src[k+1]; k+=2; }
      push(t, c==='_'?{subScript:true}:{superScript:true}); continue; }
    let e=k; while(e<src.length && src[e]!=='_' && src[e]!=='^') e++;
    push(src.slice(k,e),{}); k=e; }
  return out; }
function runs(text,base={}){
  const res=[]; const re=/(\*\*[^*]+\*\*|\$[^$]+\$|\*[^*\s][^*]*\*|\[TBD[^\]]*\]|\[Author[^\]]*\])/g; let last=0,m;
  while((m=re.exec(text))){ if(m.index>last) res.push(new TextRun({text:text.slice(last,m.index),font:F,...base}));
    const s=m[0];
    if(s.startsWith('**')) res.push(...runs(s.slice(2,-2),{...base,bold:true}));
    else if(s.startsWith('$')) res.push(...mathRuns(s.slice(1,-1),base));
    else if(s.startsWith('*')) res.push(new TextRun({text:s.slice(1,-1),italics:true,font:F,...base}));
    else if(s.startsWith('[TBD')) res.push(new TextRun({text:s,font:F,...base}));
    else res.push(new TextRun({text:s,font:F,color:'C00000',italics:true,...base}));
    last=m.index+s.length;}
  if(last<text.length) res.push(new TextRun({text:text.slice(last),font:F,...base}));
  return res;}
const P=(o)=>new Paragraph(o);
const body=[]; let i=0;
function pngSize(f){const b=fs.readFileSync(f);return [b.readUInt32BE(16),b.readUInt32BE(20)];}
const W=9360;
while(i<L.length){
  let l=L[i];
  if(l.startsWith('!TITLE ')) body.push(P({alignment:AlignmentType.CENTER,spacing:{after:240},children:[new TextRun({text:l.slice(7),bold:true,size:32,font:F})]}));
  else if(l.startsWith('!CENTER ')) body.push(P({alignment:AlignmentType.CENTER,children:runs(l.slice(8))}));
  else if(l.startsWith('# ')) body.push(P({heading:HeadingLevel.HEADING_1,children:[new TextRun({text:l.slice(2),font:F})]}));
  else if(l.startsWith('## ')) body.push(P({heading:HeadingLevel.HEADING_2,children:[new TextRun({text:l.slice(3),font:F})]}));
  else if(l.startsWith('- ')) body.push(P({numbering:{reference:'b',level:0},children:runs(l.slice(2))}));
  else if(l.startsWith('!NOTE ')) body.push(P({shading:{type:ShadingType.CLEAR,fill:'FDECEA'},children:[new TextRun({text:l.slice(6),font:F,color:'C00000',italics:true,size:20})]}));
  else if(l.startsWith('!EQ ')) body.push(P({alignment:AlignmentType.CENTER,spacing:{before:120,after:120},children:[new TextRun({text:l.slice(4),font:'Cambria Math'})]}));
  else if(l.startsWith('!REF ')) body.push(P({indent:{left:480,hanging:480},spacing:{after:60,line:276},children:runs(l.slice(5),{size:22})}));
  else if(l.startsWith('!CAP ')) body.push(P({alignment:AlignmentType.CENTER,spacing:{after:240,line:276},children:[new TextRun({text:l.slice(5),italics:true,font:F,size:20})]}));
  else if(l.startsWith('!PLACEHOLDER ')) body.push(P({alignment:AlignmentType.CENTER,border:{top:{style:BorderStyle.DASHED,size:6,color:'C00000'},bottom:{style:BorderStyle.DASHED,size:6,color:'C00000'},left:{style:BorderStyle.DASHED,size:6,color:'C00000'},right:{style:BorderStyle.DASHED,size:6,color:'C00000'}},spacing:{before:240,after:120},children:[new TextRun({text:'[FIGURE PLACEHOLDER] '+l.slice(13),font:F,color:'C00000',bold:true})]}));
  else if(l.startsWith('!MATH ')){
    const path=l.slice(6).trim(); const [w,h]=pngSize(path); const ww=Math.min(600,Math.round(w*96/300*1.09)), hh=Math.round(ww*h/w);
    body.push(P({alignment:AlignmentType.CENTER,spacing:{before:120,after:120,line:240,lineRule:d.LineRuleType.AUTO},children:[new ImageRun({type:'png',data:fs.readFileSync(path),transformation:{width:ww,height:hh}})]}));}
  else if(l.startsWith('!FIG ')){
    const [path,cap]=l.slice(5).split(' | '); const [w,h]=pngSize(path); const ww=path.includes('membership')?400:600, hh=Math.round(ww*h/w);
    body.push(P({alignment:AlignmentType.CENTER,keepNext:true,spacing:{before:200,line:240,lineRule:d.LineRuleType.AUTO},children:[new ImageRun({type:'png',data:fs.readFileSync(path),transformation:{width:ww,height:hh}})]}));
    body.push(P({alignment:AlignmentType.CENTER,spacing:{after:240,line:276},children:[new TextRun({text:cap,italics:true,font:F,size:20})]}));}
  else if(l==='!CODE'){ i++; while(L[i]!=='!END'){ body.push(P({spacing:{line:240,after:0},shading:{type:ShadingType.CLEAR,fill:'F5F5F5'},children:[new TextRun({text:L[i]||' ',font:'Courier New',size:17})]})); i++; } body.push(P({children:[]})); }
  else if(l==='!TABLE'){ i++; const rows=[]; while(L[i]!=='!END'){ rows.push(L[i].split(' | ')); i++; }
    const n=rows[0].length; const cw=Array(n).fill(Math.floor(W/n)); cw[n-1]=W-cw[0]*(n-1);
    if(n===3){cw[0]=2000;cw[1]=1100;cw[2]=W-3100;} if(n===5&&rows[0][0]==='Rule'){cw.splice(0,5,800,2300,1100,900,W-5100);}
    const brd={style:BorderStyle.SINGLE,size:4,color:'999999'};
    body.push(new Table({width:{size:W,type:WidthType.DXA},columnWidths:cw,rows:rows.map((r,ri)=>new TableRow({tableHeader:ri===0,children:r.map((c,ci)=>new TableCell({width:{size:cw[ci],type:WidthType.DXA},borders:{top:brd,bottom:brd,left:brd,right:brd},shading:ri===0?{type:ShadingType.CLEAR,fill:'E8EEF7'}:undefined,margins:{top:40,bottom:40,left:80,right:80},children:[P({spacing:{line:240,after:0},children:runs(c,{size:19,bold:ri===0})})]}))}))})); }
  else if(l.trim()==='') {}
  else body.push(P({alignment:AlignmentType.JUSTIFIED,children:runs(l)}));
  i++;}
const doc=new Document({
  styles:{default:{document:{run:{font:F,size:24},paragraph:{spacing:{line:360,lineRule:d.LineRuleType.AUTO,after:120}}}},
   paragraphStyles:[
    {id:'Heading1',name:'Heading 1',basedOn:'Normal',next:'Normal',quickFormat:true,run:{font:F,size:28,bold:true},paragraph:{spacing:{before:300,after:120},keepNext:true,outlineLevel:0}},
    {id:'Heading2',name:'Heading 2',basedOn:'Normal',next:'Normal',quickFormat:true,run:{font:F,size:24,bold:true,italics:true},paragraph:{spacing:{before:200,after:100},keepNext:true,outlineLevel:1}}]},
  numbering:{config:[{reference:'b',levels:[{level:0,format:LevelFormat.BULLET,text:'•',alignment:AlignmentType.LEFT,style:{paragraph:{indent:{left:720,hanging:360}}}}]}]},
  sections:[{properties:{page:{size:{width:11906,height:16838},margin:{top:1270,bottom:1270,left:1270,right:1270}},lineNumbers:{countBy:1,restart:'continuous'}},children:body}]});
Packer.toBuffer(doc).then(b=>{fs.writeFileSync(out,b);console.log('ok',out)});
