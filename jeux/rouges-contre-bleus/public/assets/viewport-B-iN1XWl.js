import{n as e,t}from"./three-webgl-fk9L23ZP.js";import{a as n,d as r,f as i,n as a,t as o}from"./world-GLCk6m0L.js";import{$ as s,An as c,B as l,C as u,F as d,G as f,Jt as p,K as m,L as h,Mn as g,O as _,Q as v,Qt as y,R as b,S as x,Sn as S,T as C,U as w,V as T,Z as E,Zt as ee,_n as te,_t as ne,a as re,c as ie,ct as ae,d as D,dn as oe,dt as se,h as O,hn as ce,i as k,j as le,jn as A,l as j,ln as ue,mn as de,nn as fe,nt as M,pn as pe,r as N,s as me,sn as he,un as P,ut as F,v as ge,vn as _e,w as ve,z as ye}from"./three-core-DXi6C2Cj.js";import{c as be,d as I,h as xe,i as Se,l as Ce,o as we,r as L}from"./client-CUWpWFpY.js";var Te=3.2,Ee=24*Math.PI/180,De=56*Math.PI/180,Oe=(e,t,n)=>e<t?t:e>n?n:e;function ke(e){let t=Oe(Math.log(e/Te)/Math.log(340/Te),0,1),n=t*t*(3-2*t);return Ee+(De-Ee)*(.35*t+.65*n)}function Ae(e,t=0){let n=ke(e.distance),r={x:e.x,y:t+Ne(e.distance),z:e.y};return{eye:{x:r.x,y:r.y+Math.sin(n)*e.distance,z:r.z+Math.cos(n)*e.distance},target:r}}function je(e,t,n,r=340){return{x:Oe(e.x,0,t),y:Oe(e.y,0,n),distance:Oe(e.distance,Te,r)}}function Me(e,t,n=0){if(t.y>=-1e-6)return null;let r=(n-e.y)/t.y;return r<0?null:{x:e.x+t.x*r,y:e.z+t.z*r}}function Ne(e){return Math.min(1,e*.12)}function Pe(e,t,n,r,i=38){let a=ke(e),o={x:0,y:Math.sin(a)*e,z:Math.cos(a)*e},s=-Math.sin(a),c=-Math.cos(a),l=-c,u=Math.hypot(l,0)||1,d=l/u,f=0/u,p=-f*s,m=f*0-d*c,h=d*s,g=Math.tan(i*Math.PI/360),_=n*g*t,v=r*g;return{eye:o,dir:{x:0+d*_+p*v,y:s+m*v,z:c+f*_+h*v}}}function Fe(e,t,n,r,i,a=()=>0,o=38){let{eye:s,dir:c}=Pe(e,t,n,r,o),l=Ne(e),u=i.x,d=i.y;if(c.y>=-1e-6)return{x:u,y:d};for(let e=0;e<4;e++){let e=a(u,d)+l+s.y,t=(i.h-e)/c.y;u=i.x-s.x-c.x*t,d=i.y-s.z-c.z*t}return{x:u,y:d}}function Ie(e,t,n,r,i=0,a=38){let{eye:o,target:s}=Ae(e),c=s.x-o.x,l=s.y-o.y,u=s.z-o.z,d=Math.hypot(c,l,u);c/=d,l/=d,u/=d;let f=-u,p=c,m=Math.hypot(f,p)||1;f/=m,p/=m;let h=-p*l,g=p*c-f*u,_=f*l,v=n-o.x,y=i-o.y,b=r-o.z,x=v*c+y*l+b*u,S=Math.tan(a*Math.PI/360);return{x:(v*f+b*p)/(x*S*t),y:(v*h+y*g+b*_)/(x*S),depth:x}}function Le(e,t,n,r,i=38,a=.97,o=a){let s=s=>{let c={x:t,y:n,distance:s};return e.every(({x:e,y:t})=>{let n=Ie(c,r,e,t,0,i);return n.depth>0&&Math.abs(n.x)<=a&&Math.abs(n.y)<=o})};if(!s(340))return 340;let c=Te,l=340;for(let e=0;e<40;e++){let e=(c+l)/2;s(e)?l=e:c=e}return l}function Re(e,t,n,r,i,a,o=38){let s=Math.max(.5,n/2);return Le(Array.from({length:24},(n,r)=>{let i=r/24*Math.PI*2;return{x:e+Math.cos(i)*s,y:t+Math.sin(i)*s}}),e,t,r,o,i,a)}function ze(e,t){return 1-Math.exp(-e*t)}function Be(e,t){return{x:t*(Math.sin(e*47.1)*.6+Math.sin(e*83.7)*.4),y:t*Math.sin(e*61.3)*.5,z:t*(Math.sin(e*39.9)*.6+Math.sin(e*71.9)*.4)}}var Ve=class{mapWidth;mapHeight;current;goal;shake=0;clock=0;maxDistance=340;constructor(e,t,n){this.mapWidth=t,this.mapHeight=n,this.current=je(e,t,n),this.goal={...this.current}}setGoal(e,t=!1){this.goal=je(e,this.mapWidth,this.mapHeight,this.maxDistance),t&&(this.current={...this.goal})}pan(e,t,n=!0){this.goal=je({...this.goal,x:this.goal.x+e,y:this.goal.y+t},this.mapWidth,this.mapHeight,this.maxDistance),n&&(this.current=je({...this.current,x:this.current.x+e,y:this.current.y+t},this.mapWidth,this.mapHeight,this.maxDistance))}addShake(e){this.shake=Math.min(1.5,this.shake+e)}update(e){this.clock+=e;let t=ze(9,e),n=ze(7,e),r=this.current,i=this.goal,a=Math.log(r.distance)+(Math.log(i.distance)-Math.log(r.distance))*n;this.current={x:r.x+(i.x-r.x)*t,y:r.y+(i.y-r.y)*t,distance:Math.exp(a)},this.shake*=Math.exp(-e*5),this.shake<.001&&(this.shake=0)}shakeNow(){return this.shake===0?{x:0,y:0,z:0}:Be(this.clock,this.shake*(.05+this.current.distance*.006))}},R=[3042288,15217963],z=[1785754,10231064],B=[9417983,16751247],He=[8364006,15243916],V={skin:15911328,trousers:15854816,boots:2827810,belt:3812902,metal:14278114,wood:9067059,iron:4869975,gold:15909424,bronze:7297594,straw:15256958,canvas:15326400,horse:9723962,mane:3811871},H={grass:11848074,grassDark:10335863,grassLight:13096860,meadow:11125379,road:14733219,roadEdge:13153155,sand:15128235,earth:12164983,rock:10986652,rockDark:8815484,stone:14209732,stoneDark:11774874,roofNeutral:9407104,wood:10120002,woodDark:7227948,waterDeep:6262675,waterShallow:10273723,foam:15266022,trunk:8017462,leaf:9416810,leafDark:7312214,leafLight:11125884,pine:6261338,skyTop:9418713,skyHorizon:15328470,fog:14672856,rope:15721410,plank:13212258},Ue=[`+50`,`×2`,`×3`,`÷2`],U={heal:10155460,healDeep:3124857,canvas:15985884,gold:16763194,goldLight:16773552,goldDeep:13208082,ember:16747050,emberLight:16765050,forbidden:16734799},We=[`I`,`II`,`III`];function W(e){return`#${e.toString(16).padStart(6,`0`)}`}var Ge={sun:16773596,sunIntensity:2.9,sky:14477813,ground:9076578,hemiIntensity:1.55,fog:H.fog,fogDensity:.0021,skyTop:H.skyTop,skyHorizon:H.skyHorizon},Ke={sun:16757370,sunIntensity:2.5,sky:9411286,ground:6179396,hemiIntensity:1.3,fog:12163212,fogDensity:.0026,skyTop:2964076,skyHorizon:15770730},qe=3.2;function Je(e,t=qe){if(!(t>0))return 1;let n=Math.max(0,Math.min(1,e/t));return n*n*(3-2*n)}var Ye=new j,Xe=new j;function Ze(e,t,n,r){return Ye.setHex(e),Xe.setHex(t),r.copy(Ye).lerp(Xe,n)}function Qe(e,t,n){let r=Math.max(0,Math.min(1,n)),i=new j,a=(e,t)=>Ze(e,t,r,i).getHex();return{sun:a(e.sun,t.sun),sunIntensity:e.sunIntensity+(t.sunIntensity-e.sunIntensity)*r,sky:a(e.sky,t.sky),ground:a(e.ground,t.ground),hemiIntensity:e.hemiIntensity+(t.hemiIntensity-e.hemiIntensity)*r,fog:a(e.fog,t.fog),fogDensity:e.fogDensity+(t.fogDensity-e.fogDensity)*r,skyTop:a(e.skyTop,t.skyTop),skyHorizon:a(e.skyHorizon,t.skyHorizon)}}var $e=class{lights;renderer;k=0;started=-1;skyMaterial;vignette;flashEl;flash=0;constructor(e,t,n,r){this.lights=e,this.renderer=n,this.skyMaterial=t.material;let i=e=>{let t=document.createElement(`div`);return Object.assign(t.style,{position:`absolute`,inset:`0`,pointerEvents:`none`,opacity:`0`,background:e}),r.appendChild(t),t};this.vignette=i(`radial-gradient(ellipse at 50% 45%, rgba(0,0,0,0) 45%, rgba(60,14,4,0.55) 100%)`),this.flashEl=i(`radial-gradient(ellipse at 50% 50%, rgba(255,236,200,0.95) 0%, rgba(255,140,60,0.75) 60%, rgba(120,30,10,0.6) 100%)`)}begin(e){this.started<0&&(this.started=e),this.flash=1}update(e,t,n){e&&this.started<0&&(this.started=t-qe);let r=this.started>=0?Je(t-this.started):0;Math.abs(r-this.k)>1e-4&&(this.k=r,this.apply(this.k)),this.flash>0&&(this.flash=Math.max(0,this.flash-n/1.1),this.flashEl.style.opacity=(this.flash*this.flash*.85).toFixed(3))}apply(e){let t=this.lights,n=Qe(Ge,Ke,e);t.sun.color.setHex(n.sun),t.sun.intensity=n.sunIntensity,t.hemi.color.setHex(n.sky),t.hemi.groundColor.setHex(n.ground),t.hemi.intensity=n.hemiIntensity,t.fog.color.setHex(n.fog),t.fog.density=n.fogDensity,t.sync();let r=this.skyMaterial.uniforms;r.uTop.value.setHex(n.skyTop),r.uHorizon.value.setHex(n.skyHorizon),r.uFog.value.setHex(n.fog),this.renderer.setClearColor(n.fog),this.vignette.style.opacity=(e*.9).toFixed(3)}dispose(){this.vignette.remove(),this.flashEl.remove()}},et=8,tt=56;function nt(e=et,t=tt){let n=1+e*t,r=new Float32Array(n*3),i=new Float32Array(n*2);for(let n=1;n<=e;n++){let r=n/e;for(let e=0;e<t;e++){let a=e/t*Math.PI*2,o=1+(n-1)*t+e;i[o*2]=Math.cos(a)*r,i[o*2+1]=Math.sin(a)*r}}let a=[];for(let e=0;e<t;e++)a.push(0,1+(e+1)%t,1+e);for(let n=1;n<e;n++){let e=1+(n-1)*t,r=1+n*t;for(let n=0;n<t;n++){let i=(n+1)%t;a.push(e+n,e+i,r+n,e+i,r+i,r+n)}}let o=new re;return o.setAttribute(`position`,new k(r,3)),o.setAttribute(`aP`,new k(i,2)),o.setIndex(a),o}function rt(e,t,n,r,i,a,o=1/0){let s=e.attributes.position,c=e.attributes.aP,l=s.array,u=c.array,d=i(t,n)+o;for(let e=0;e<s.count;e++){let o=t+u[e*2]*r,s=n+u[e*2+1]*r;l[e*3]=o,l[e*3+1]=Math.min(d,i(o,s))+a,l[e*3+2]=s}s.needsUpdate=!0,e.computeBoundingSphere()}var it=`
in vec2 aP;
out vec2 vP;
void main() {
  vP = aP;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;function at(e,t,n){let r=e.index?e.toNonIndexed():e;r!==e&&e.dispose(),n&&r.applyMatrix4(n);for(let e of Object.keys(r.attributes))e!==`position`&&r.deleteAttribute(e);let i=new j(t),a=r.attributes.position.count,o=new Float32Array(a*3);for(let e=0;e<a;e++)o[e*3]=i.r,o[e*3+1]=i.g,o[e*3+2]=i.b;return r.setAttribute(`color`,new k(o,3)),r}var G=class{parts=[];add(e,t,n){return this.parts.push(at(e,t,n)),this}get empty(){return this.parts.length===0}build(){let e=t(this.parts,!1);for(let e of this.parts)e.dispose();if(this.parts.length=0,!e)throw Error(`kit: nothing to merge`);return e.computeVertexNormals(),e.computeBoundingSphere(),e}};function K(e,t,n,r=0,i=1,a=i,o=i,s=0,c=0){let l=new se().setFromEuler(new C(s,r,c,`YXZ`));return new E().compose(new A(e,t,n),l,new A(i,a,o))}function ot(){return new M({vertexColors:!0,flatShading:!0})}function st(e){let t=e>>>0;return()=>{t=t+1831565813>>>0;let e=t;return e=Math.imul(e^e>>>15,e|1),e^=e+Math.imul(e^e>>>7,e|61),((e^e>>>14)>>>0)/4294967296}}function ct(e,t){let n=(e,t)=>{let n=Math.imul(e,374761393)+Math.imul(t,668265263)|0;return n=Math.imul(n^n>>>13,1274126177),((n^n>>>16)>>>0)/4294967296},r=Math.floor(e),i=Math.floor(t),a=e-r,o=t-i,s=a*a*(3-2*a),c=o*o*(3-2*o),l=n(r,i)+(n(r+1,i)-n(r,i))*s;return l+(n(r,i+1)+(n(r+1,i+1)-n(r,i+1))*s-l)*c}var lt=[[-.3,.46],[-.26,.05],[-.18,-.25],[-.1,-.42],[-.02,-.5],[.02,-.36],[.2,-.22],[.44,.02],[.44,.14],[.3,.18],[.1,.06],[.12,.46]];function ut(e,t){if(typeof document<`u`){let n=document.createElement(`canvas`);return n.width=e,n.height=t,n}return new OffscreenCanvas(e,t)}var dt=class{width;height;canvas;ctx;texture;key=``;constructor(e,t){this.width=e,this.height=t,this.canvas=ut(e,t),this.ctx=this.canvas.getContext(`2d`),this.texture=new me(this.canvas),this.texture.colorSpace=he,this.texture.minFilter=f,this.texture.generateMipmaps=!1,this.texture.anisotropy=4}update(e,t){e!==this.key&&(this.key=e,this.ctx.clearRect(0,0,this.width,this.height),t(this.ctx),this.texture.needsUpdate=!0)}dispose(){this.texture.dispose()}};function q(e,t,n,r,i,a){e.beginPath(),e.moveTo(t+a,n),e.arcTo(t+r,n,t+r,n+i,a),e.arcTo(t+r,n+i,t,n+i,a),e.arcTo(t,n+i,t,n,a),e.arcTo(t,n,t+r,n,a),e.closePath()}var ft=`"Avenir Next", "Segoe UI", system-ui, sans-serif`;function pt(e,t,n,r,i,a,o,s,c,l=!1){q(e,6,6,t-12,n-12,18),e.fillStyle=l?`rgba(255,255,255,0.6)`:W(s),e.fill(),e.lineWidth=6,e.strokeStyle=l?`rgba(40,40,40,0.6)`:`#fdf6e3`,e.stroke(),e.fillStyle=l?`#333`:`#ffffff`,e.textAlign=`center`,e.textBaseline=`middle`;let u=Ue[r]??`?`,d=a>0;if(e.font=`800 ${d?66:80}px ${ft}`,e.shadowColor=l?`transparent`:W(c),e.shadowOffsetY=3,e.fillText(u,d?t*.33:t/2,n*.47),d&&(e.font=`700 46px ${ft}`,e.fillText(String(Math.max(0,Math.ceil(i))),t*.7,n*.47)),e.shadowColor=`transparent`,l&&(e.strokeStyle=`#d23b2f`,e.lineWidth=10,e.lineCap=`round`,e.beginPath(),e.moveTo(t*.16,n*.8),e.lineTo(t*.84,n*.2),e.stroke()),o!==null&&o<.999){let r=t-60;e.fillStyle=`rgba(0,0,0,0.45)`,e.fillRect(30,n-26,r,10),e.fillStyle=o>.5?`#bff08a`:o>.25?`#ffd34d`:`#ff8a5c`,e.fillRect(30,n-26,r*Math.max(0,o),10)}}function mt(e,t,n,r,i){if(e.save(),e.translate(n,r),e.strokeStyle=`#ffffff`,e.fillStyle=`#ffffff`,e.lineCap=`round`,e.lineJoin=`round`,e.lineWidth=i*.1,e.beginPath(),t===Ce.Sword)e.rotate(-Math.PI/4),e.moveTo(0,i*.42),e.lineTo(0,-i*.45),e.moveTo(-i*.18,i*.2),e.lineTo(i*.18,i*.2),e.stroke();else if(t===Ce.Spear)e.rotate(-Math.PI/4),e.moveTo(0,i*.5),e.lineTo(0,-i*.3),e.stroke(),e.beginPath(),e.moveTo(0,-i*.52),e.lineTo(i*.11,-i*.28),e.lineTo(-i*.11,-i*.28),e.closePath(),e.fill();else if(t===Ce.Bow){e.arc(-i*.15,0,i*.42,-Math.PI/2.6,Math.PI/2.6),e.stroke(),e.lineWidth=i*.04,e.beginPath();let t=Math.PI/2.6;e.moveTo(-i*.15+Math.cos(t)*i*.42,Math.sin(t)*i*.42),e.lineTo(-i*.15+Math.cos(t)*i*.42,-Math.sin(t)*i*.42),e.stroke()}else if(t===Ce.Cavalry){for(let[t,n]of lt)e.lineTo(t*i,n*i);e.closePath(),e.fill()}else e.arc(-i*.08,i*.18,i*.2,0,Math.PI*2),e.stroke(),e.lineWidth=i*.16,e.beginPath(),e.moveTo(-i*.1,i*.05),e.lineTo(i*.42,-i*.22),e.stroke();e.restore()}function ht(e,t,n,r,i,a,o){e.beginPath(),e.moveTo(8,8),e.lineTo(t-8,8),e.lineTo(t-8,n-30),e.lineTo(t/2,n-6),e.lineTo(8,n-30),e.closePath(),e.fillStyle=o?`#8b8780`:W(a),e.fill(),e.lineWidth=6,e.strokeStyle=`#fdf6e3`,e.stroke(),mt(e,r,t/2,n*.36,t*.62),e.fillStyle=i>0?`#ffd84a`:`#ffffff`,e.font=`800 34px ${ft}`,e.textAlign=`center`,e.textBaseline=`middle`,e.fillText(We[i]??``,t/2,n*.72)}function gt(e,t,n,r){e.save(),e.translate(t,n),e.rotate(-Math.PI/4),e.fillStyle=`#ffffff`,q(e,-r*.07,-r*.22,r*.14,r*.72,r*.05),e.fill(),q(e,-r*.34,-r*.46,r*.68,r*.26,r*.06),e.fill(),e.restore()}function _t(e,t,n,r){e.save(),e.translate(t,n),e.beginPath(),e.moveTo(0,-r*.46),e.lineTo(r*.5,r*.4),e.lineTo(-r*.5,r*.4),e.closePath(),e.lineJoin=`round`,e.lineWidth=r*.1,e.strokeStyle=`#3a3226`,e.stroke(),e.fillStyle=`#ffd34d`,e.fill(),e.fillStyle=`#3a3226`,e.font=`900 ${Math.round(r*.62)}px ${ft}`,e.textAlign=`center`,e.textBaseline=`middle`,e.fillText(`!`,0,r*.08),e.restore()}function vt(e,t,n,r,i,a,o){q(e,6,6,t-12,n-12,22),e.fillStyle=o?`#8b8780`:W(i),e.fill(),e.lineWidth=6,e.strokeStyle=`#fdf6e3`,e.stroke();let s=n*.62;o?_t(e,n*.52,n/2,s):gt(e,n*.5,n/2+2,s);let c=Math.max(0,Math.min(1,r)),l=n*.98,u=t-l-26;e.fillStyle=`#ffffff`,e.textAlign=`center`,e.textBaseline=`middle`,e.font=`800 ${Math.round(n*.36)}px ${ft}`,e.shadowColor=o?`transparent`:W(a),e.shadowOffsetY=3,e.fillText(`${Math.floor(c*100)} %`,l+u/2,n*.36),e.shadowColor=`transparent`;let d=n*.62,f=n*.17;q(e,l,d,u,f,f/2),e.fillStyle=`rgba(0,0,0,0.38)`,e.fill(),c>0&&(q(e,l,d,Math.max(f,u*c),f,f/2),e.fillStyle=o?`#d8d3c8`:`#ffd84a`,e.fill())}function yt(e,t,n,r,i,a){e.save(),e.translate(t,n+r*.04),e.beginPath(),e.moveTo(0,r*.42),e.bezierCurveTo(-r*.62,-r*.02,-r*.32,-r*.52,0,-r*.2),e.bezierCurveTo(r*.32,-r*.52,r*.62,-r*.02,0,r*.42),e.closePath(),a&&(e.lineWidth=r*.1,e.lineJoin=`round`,e.strokeStyle=a,e.stroke()),e.fillStyle=i,e.fill(),e.strokeStyle=a??`rgba(0,0,0,0.35)`,e.lineWidth=r*.07,e.lineCap=`round`,e.lineJoin=`round`,e.beginPath(),e.moveTo(-r*.3,r*.02),e.lineTo(-r*.1,r*.02),e.lineTo(-r*.02,-r*.14),e.lineTo(r*.08,r*.16),e.lineTo(r*.15,r*.02),e.lineTo(r*.3,r*.02),e.stroke(),e.restore()}function bt(e,t,n,r,i,a,o,s,c,l=!1){q(e,6,6,t-12,n-12,22),e.fillStyle=l?`rgba(255,255,255,0.72)`:W(o),e.fill(),e.lineWidth=6,e.strokeStyle=l?`rgba(40,40,40,0.55)`:`#fdf6e3`,e.stroke(),yt(e,n*.52,n*.5,n*.66,l?`#4c8f6f`:`#ffffff`,l?void 0:W(s));let u=i>0?Math.max(0,Math.min(1,r/i)):0,d=n*.98,f=t-d-26;e.fillStyle=l?`#333`:`#ffffff`,e.textAlign=`center`,e.textBaseline=`middle`,e.font=`800 ${Math.round(n*.38)}px ${ft}`,e.shadowColor=l?`transparent`:W(s),e.shadowOffsetY=3,e.fillText(String(Math.max(0,Math.ceil(r))),d+f/2,n*.36),e.shadowColor=`transparent`;let p=n*.6,m=n*.16;if(q(e,d,p,f,m,m/2),e.fillStyle=`rgba(0,0,0,0.38)`,e.fill(),u>0&&(q(e,d,p,Math.max(m,f*u),m,m/2),e.fillStyle=W(c),e.fill()),l===`forbidden`&&(e.strokeStyle=`#d23b2f`,e.lineWidth=10,e.lineCap=`round`,e.beginPath(),e.moveTo(t*.12,n*.82),e.lineTo(t*.88,n*.18),e.stroke()),a!==null&&a<.999){let r=t-12-36,i=n-6-13;e.fillStyle=`rgba(0,0,0,0.5)`,e.fillRect(24,i,r,8),e.fillStyle=a>.5?`#bff08a`:a>.25?`#ffd34d`:`#ff8a5c`,e.fillRect(24,i,r*Math.max(0,a),8)}}function xt(e,t,n,r,i,a,o,s){q(e,8,8,t-16,n-16,22);let c=e.createLinearGradient(0,8,0,n-8);c.addColorStop(0,`#fff3b8`),c.addColorStop(.45,`#ffcf3d`),c.addColorStop(1,`#d99413`),e.fillStyle=c,e.fill(),e.lineWidth=s===null?6:12,e.strokeStyle=s===null?`#fffaf0`:W(s),e.stroke(),e.textAlign=`center`,e.textBaseline=`middle`,e.fillStyle=`#ffffff`,e.shadowColor=`#9a5d05`,e.shadowOffsetY=4,e.font=`900 70px ${ft}`,e.fillText(r,t*.33,n*.46),e.font=`800 44px ${ft}`;let l=o===null?String(Math.max(0,Math.ceil(i))):`${Math.max(0,Math.ceil(o))} s`;if(e.fillText(l,t*.71,n*.46),e.shadowColor=`transparent`,o===null&&a>0){let r=Math.max(0,Math.min(1,i/a)),o=t-70;q(e,35,n-30,o,10,5),e.fillStyle=`rgba(90,50,0,0.45)`,e.fill(),r>0&&(q(e,35,n-30,Math.max(10,o*r),10,5),e.fillStyle=`#ffffff`,e.fill())}}var St=.6;function Ct(e,t){return!(t>0)||!(e>0)?0:Math.min(1,Math.sqrt(e/t/St))}function wt(e,t=7){let n=st(t),r=e*.625,i=e*.62,a=.035,o=[];for(let[e,t]of[[1,0],[-1,0],[0,1],[0,-1]])o.push({at:(n,i)=>[e*(r+a)+t*n,i,t*(r+a)+e*n],normal:()=>[e,0,t],u:[-i+1.7,i-1.7],top:5.8,bottom:.7});let s=(e,t,n,r,i)=>({at:(i,o)=>[e+Math.cos(r+i)*(n+a),o,t+Math.sin(r+i)*(n+a)],normal:e=>[Math.cos(r+e),0,Math.sin(r+e)],u:[-.9,.9],top:i,bottom:1});for(let[e,t]of[[i,i],[i,-i],[-i,i],[-i,-i]])o.push(s(e,t,1.52,Math.atan2(t,e),7.6));for(let e=0;e<4;e++)o.push(s(0,0,2.28,e/4*Math.PI*2+.4,11.4));let c=[],l=(e,t,r,i,a,o)=>{let s=r,u=i,d=t,f=e.u[1]-e.u[0];for(let t=0;t<a&&u>e.bottom;t++){let t=(n()-.5)*f*.16,r=.35+n()*.45,i=Math.max(e.u[0],Math.min(e.u[1],s+t)),a=u-r;c.push({threshold:d,a:e.at(s,u),b:e.at(i,a),n:e.normal(s,u),width:o}),n()<.18&&o>.06&&l(e,d+.05,i,a,3+Math.floor(n()*3),o*.6),s=i,u=a,d+=.035+n()*.03,o*=.93}},u=[];for(let e of o){let t=e.top>8?1:2;for(let r=0;r<t;r++)u.push([e,n()])}return u.sort((e,t)=>e[1]-t[1]),u.forEach(([e,t],r)=>{let i=r/u.length*.7,a=e.u[0]+(e.u[1]-e.u[0])*(.2+.6*t),o=e.top-n()*.6;l(e,i,a,o,6+Math.floor(n()*6),.13+n()*.05)}),c.sort((e,t)=>e.threshold-t.threshold),c}function Tt(e,t){let n=wt(e,t),r=new Float32Array(n.length*6*3),i=[],a=[];n.forEach((e,t)=>{let n=e.b[0]-e.a[0],o=e.b[1]-e.a[1],s=e.b[2]-e.a[2],c=e.n[1]*s-e.n[2]*o,l=e.n[2]*n-e.n[0]*s,u=e.n[0]*o-e.n[1]*n,d=Math.hypot(c,l,u)||1;c=c/d*e.width*.5,l=l/d*e.width*.5,u=u/d*e.width*.5;let f=[[e.a[0]-c,e.a[1]-l,e.a[2]-u],[e.a[0]+c,e.a[1]+l,e.a[2]+u],[e.b[0]+c*.7,e.b[1]+l*.7,e.b[2]+u*.7],[e.b[0]-c*.7,e.b[1]-l*.7,e.b[2]-u*.7]],p=t*18;for(let e of[0,1,2,0,2,3])r[p++]=f[e][0],r[p++]=f[e][1],r[p++]=f[e][2];i.push(e.threshold),a.push((t+1)*6)});let o=new re;return o.setAttribute(`position`,new k(r,3)),o.computeBoundingSphere(),o.setDrawRange(0,0),{geometry:o,thresholds:i,ends:a}}var Et={x:5.2,z:4.2},Dt=5.6,Ot=256,kt=96,At=5.6,jt=.6;function Mt(e,t,n){let r=t>0?Math.max(0,Math.min(1,e/t)):1;return n||r>0?Math.max(Nt,r):0}var Nt=.001;function Pt(e,t){let n=0,r=e.length;for(;n<r;){let i=n+r>>1;e[i]<=t?n=i+1:r=i}return n}function Ft(e){let t=[],n=(e,n,r,i=K(0,0,0))=>{t.push({threshold:e,geometry:at(n,r,i)})},r=R[e],i=Et.x/2+.35,a=Et.z/2+.35;n(0,new N(2*i,.05,2*a),15129531,K(0,.025,0));for(let[e,t]of[[i,a],[i,-a],[-i,a],[-i,-a]])n(0,new N(.1,.85,.1),H.woodDark,K(e,.42,t)),n(0,new N(.14,.16,.14),r,K(e,.86,t));for(let e of[a,-a])n(0,new N(2*i,.04,.04),H.rope,K(0,.62,e));for(let e of[i,-i])n(0,new N(.04,.04,2*a),H.rope,K(e,.62,0));n(0,new N(.07,2.3,.07),H.woodDark,K(-i,1.15,-a)),n(0,new N(.04,.55,.85),r,K(-i,1.95,-a+.45));let o=Nt;for(let e=0;e<3;e++)n(o,new N(2.2,.1,.32),H.plank,K(.3,.06+e*.1,-a-.8+e%2*.08,.05*e));for(let[e,t,r]of[[-1.4,.22,-a-.9],[-2,.22,-a-.8],[-1.7,.62,-a-.85]])n(o,new N(.5,.42,.42),H.stoneDark,K(e,t,r,.2));n(.02,new N(Et.x,.4,Et.z),H.stoneDark,K(0,.2,0));let s=2.75,c=2.25;[[-2.75,-2.25],[-2.75,c],[0,-2.25],[0,c],[s,-2.25],[s,c]].forEach(([e,t],r)=>{n(.04+r*.014,new N(.13,4.5,.13),H.woodDark,K(e,2.25,t))});let l=2.6/6,u=.35,d=.13,f=.66/6;for(let e=0;e<6;e++){let t=.4+l*(e+.5),r=e%2==0?H.stone:13354164,i=d+e*f;[[new N(4.6,l,u),0,-1.625],[new N(u,l,3.6-2*u),-2.125,0],[new N(4.6,l,u),0,1.8-u/2],[new N(u,l,3.6-2*u),2.3-u/2,0]].forEach(([e,a,o],s)=>n(i+s*f/4,e,r,K(a,t,o)))}[1.3,2.25,3.2].forEach((e,t)=>{let r=d+f*(1.6+t*2);for(let t of[-2.3,2.3])n(r,new N(5.9,.08,.5),H.plank,K(0,e,t))});for(let e of[-2.25,c])for(let[t,r]of[[-2.75/2,1],[s/2,-1]])n(.21800000000000003,new N(.07,3.1,.07),H.wood,K(t,1.7,e,0,1,1,1,0,r*.72));n(.28400000000000003,new N(.24,1.7,1.4),4930350,K(2.33,1.25,0));for(let e of[-.3,.3])n(.251,new N(.07,3.4,.07),H.wood,K(-3.2,1.6,e,0,1,1,1,0,-.2));for(let e=0;e<5;e++){let t=.4+e*.62;n(.251,new N(.05,.05,.6),H.wood,K(-3.2+.2*(t-1.6),t,0))}let p=.79+.03;return[-2,0,2].forEach((e,t)=>{for(let r of[-1,1])n(p+t*.04,new N(.13,.13,2.55),H.wood,K(e,3.85,r*.95,0,1,1,1,r*.7));n(p+t*.04,new N(.12,.12,3.9),H.wood,K(e,3.05,0))}),n(.9600000000000001,new N(4.9,.14,.14),H.woodDark,K(0,4.62,0)),t.sort((e,t)=>e.threshold-t.threshold),t}function It(e){let n=Ft(e),r=[],i=[],a=0;for(let e of n)a+=e.geometry.attributes.position.count,r.push(e.threshold),i.push(a);let o=t(n.map(e=>e.geometry),!1);for(let e of n)e.geometry.dispose();if(!o)throw Error(`sites: nothing to merge`);return o.computeVertexNormals(),o.computeBoundingSphere(),{geometry:o,thresholds:r,ends:i}}var Lt=`
out vec2 vP;
void main() {
  vP = position.xz * 2.0;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`,Rt=`
uniform vec3 uColor;
uniform float uTime;
uniform float uOpacity;
uniform float uHover;
uniform float uValid;
in vec2 vP;
void main() {
  float r = length(vP);
  if (r > 1.0) discard;
  float ring = smoothstep(0.76, 0.83, r) * (1.0 - smoothstep(0.93, 1.0, r));
  float inner = 1.0 - smoothstep(0.55, 0.8, r);
  float a;
  if (uValid > 0.5) {
    // Glowing disc, a bright pulsing rim and waves running outwards.
    float pulse = 0.5 + 0.5 * sin(uTime * (3.2 + 2.5 * uHover));
    float wave = fract(r * 1.4 - uTime * (0.55 + 0.4 * uHover));
    float band = smoothstep(0.0, 0.1, wave) * (1.0 - smoothstep(0.1, 0.28, wave)) * inner;
    a = inner * (0.2 + 0.14 * uHover)
      + ring * (0.62 + 0.3 * pulse + 0.2 * uHover)
      + band * (0.28 + 0.22 * uHover);
  } else {
    // Forbidden: dashed rim and diagonal hatching, no pulse.
    float ang = atan(vP.y, vP.x);
    float dash = step(0.45, fract(ang * 3.8197));
    float hatch = step(0.6, fract((vP.x + vP.y) * 3.2)) * inner;
    a = ring * dash * (0.7 + 0.25 * uHover) + hatch * (0.2 + 0.14 * uHover) + inner * 0.06;
  }
  gl_FragColor = vec4(uColor, min(0.95, a) * uOpacity);
  #include <colorspace_fragment>
}
`,zt=16769658,Bt=16734799,Vt=class{heightAt;group=new d;sites=[[],[]];highlightGeometry=new F(1,1).rotateX(-Math.PI/2);proxyGeometry=new N(Et.x+1.4,5,Et.z+1.4).translate(0,2.5,0);proxyMaterial=new s({visible:!1});initialised=!1;shadowDirty=!1;lastShadow=-1/0;onChange=null;constructor(e,t){this.heightAt=t,this.group.name=`sites`;for(let t of[0,1])(e.sites?.[t]??[]).forEach((e,n)=>{this.sites[t].push(this.build(t,n,e.x,e.y))})}build(e,t,n,r){let i=new d;i.name=`site-${e}-${t}`;let a=this.heightAt(n,r);i.position.set(n,a,r);let o=It(e),s=new M({vertexColors:!0,flatShading:!0}),c=new v(o.geometry,s);c.castShadow=!0,c.rotation.y=e===0?0:Math.PI;let l=Pt(o.thresholds,0);o.geometry.setDrawRange(0,o.ends[l-1]??0);let u=new dt(Ot,kt),f=new de(new ce({map:u.texture,transparent:!0,depthWrite:!1,depthTest:!1}));f.center.set(.5,0),f.renderOrder=10,f.position.set(0,Dt,0),f.visible=!1;let p=new v(this.proxyGeometry,this.proxyMaterial);p.userData.site={side:e,index:t};let m=new P({vertexShader:Lt,fragmentShader:Rt,transparent:!0,depthWrite:!1,uniforms:{uColor:{value:new j(zt)},uTime:{value:0},uOpacity:{value:0},uHover:{value:0},uValid:{value:1}}}),h=new v(this.highlightGeometry,m);return h.position.y=.12,h.renderOrder=2,h.visible=!1,i.add(c,f,p,h),this.group.add(i),{side:e,index:t,x:n,y:r,group:i,mesh:c,material:s,construction:o,plaque:f,label:u,proxy:p,highlight:h,highlightMaterial:m,hlTarget:0,hlOpacity:0,hover:0,hoverTarget:0,stage:0,drawn:l,dismantle:0,built:!1,bump:0,stalled:!1}}pickTargets(){let e=[];for(let t of this.sites)for(let n of t)n.built||e.push(n.proxy);return e}openSites(){let e=[];for(let t of this.sites)for(let n of t)n.built||e.push({side:n.side,index:n.index,x:n.x,y:n.y});return e}setHighlights(e){for(let e of this.sites)for(let t of e)t.hlTarget=0;for(let t of e??[]){let e=this.sites[t.side]?.[t.index];if(!e||e.built)continue;e.hlTarget=1,e.hoverTarget=+!!t.hovered;let n=e.highlightMaterial.uniforms;n.uValid.value=+!!t.valid,n.uColor.value.setHex(t.valid?zt:Bt)}}arrive(e,t){let n=this.sites[e]?.[t];n&&(n.bump=1)}update(e,t,n,r){let i=!this.initialised;this.initialised=!0;let a=Math.max(1,r/70);for(let o of[0,1]){let s=e.sites?.[o]??[];for(let e of this.sites[o]){let o=s[e.index];o&&this.updateSite(e,o,n,i),this.animate(e,t,n,a,r)}}this.shadowDirty&&t-this.lastShadow>=jt&&(this.shadowDirty=!1,this.lastShadow=t,this.onChange?.())}updateSite(e,t,n,r){t.barracks>=0&&!e.built&&(e.built=!0,e.hlTarget=0,r&&(e.dismantle=1));let i=Mt(t.progress,t.required,t.builder>=0||t.enRoute>0);e.stage=r||i<e.stage?i:e.stage+(i-e.stage)*(1-Math.exp(-n*2.5)),i-e.stage<.001&&(e.stage=i),e.stalled!==t.stalled&&(e.stalled=t.stalled,e.material.color.setHex(t.stalled?10263190:16777215));let a=Pt(e.construction.thresholds,e.stage);a!==e.drawn&&(e.drawn=a,e.construction.geometry.setDrawRange(0,e.construction.ends[a-1]??0),this.shadowDirty=!0);let o=!e.built&&e.stage>0;if(e.plaque.visible=o,o){let n=t.required>0?t.progress/t.required:1;e.label.update(`${Math.floor(n*100)}|${+!!t.stalled}|${t.side}`,e=>vt(e,Ot,kt,n,R[t.side],z[t.side],t.stalled))}}animate(e,t,n,r,i){e.built&&e.dismantle<1&&(e.dismantle=Math.min(1,e.dismantle+n/.9),e.dismantle>=1&&(this.shadowDirty=!0));let a=e.dismantle;e.mesh.visible=a<1,e.mesh.scale.set(1+a*.08,Math.max(.01,1-a*a),1+a*.08),e.bump=Math.max(0,e.bump-n*4);let o=1+.18*e.bump*e.bump;e.plaque.scale.set(4*r*o,1.5*r*o,1),e.plaque.position.y=Dt+Math.max(0,i-70)*.02,e.hlOpacity+=(e.hlTarget-e.hlOpacity)*(1-Math.exp(-n*12)),e.hover+=(e.hoverTarget-e.hover)*(1-Math.exp(-n*12));let s=e.hlOpacity>.01;if(e.highlight.visible=s,s){let n=e.highlightMaterial.uniforms;n.uOpacity.value=e.hlOpacity,n.uHover.value=e.hover,n.uTime.value=t;let r=2*At*Math.max(1,i/90)*(1+.08*e.hover);e.highlight.scale.set(r,1,r)}}dispose(){for(let e of this.sites)for(let t of e)t.construction.geometry.dispose(),t.material.dispose(),t.plaque.material.dispose(),t.label.dispose(),t.highlightMaterial.dispose();this.highlightGeometry.dispose(),this.proxyGeometry.dispose(),this.proxyMaterial.dispose()}},Ht=6.62;function Ut(e){let t=e*.62;return[[t,0],[-t,0],[0,t],[0,-t]]}function Wt(){let e=new D(.42,1.25,6).translate(0,.62,0).toNonIndexed(),t=e.attributes.position,n=new j(U.ember),r=new j(U.emberLight),i=new j,a=new Float32Array(t.count*3);for(let e=0;e<t.count;e++)i.copy(n).lerp(r,Math.min(1,t.getY(e)/1.1)),a.set([i.r,i.g,i.b],e*3);return e.setAttribute(`color`,new k(a,3)),e.deleteAttribute(`normal`),e.deleteAttribute(`uv`),e}var Gt=new j(3814448);function Kt(e){if(e<=0)return 0;if(e>=1)return 1;let t=e-1;return 1+3.2*t*t*t+2.2*t*t}function qt(e){let t=new G;t.add(new O(e+1.2,e+1.6,.8,8),H.stoneDark,K(0,.2,0,Math.PI/8)),t.add(new N(e*1.25,5.5,e*1.25),H.stone,K(0,3.2,0));let n=e*.62;for(let[e,r]of[[n,n],[n,-n],[-n,n],[-n,-n]])t.add(new O(1.45,1.6,8,6),H.stone,K(e,4.2,r)),t.add(new O(1.75,1.75,.5,6),H.stoneDark,K(e,8.2,r));for(let[n,r]of Ut(e))t.add(new O(.62,.36,.45,6),V.iron,K(n,Ht,r));for(let e=0;e<5;e++){let r=-n+(e+.5)*2*n/5;for(let[e,i]of[[r,n],[r,-n],[n,r],[-n,r]])t.add(new N(.8,.7,.8),H.stoneDark,K(e,6.1,i))}return t.add(new O(2.1,2.4,11,8),H.stone,K(0,6.5,0,Math.PI/8)),t.add(new N(.4,2.8,2.4),4930350,K(e*.63,1.9,0)),t.add(new N(.5,.5,3.2),H.stoneDark,K(e*.64,3.5,0)),t}function Jt(e){let t=new G,n=e*.62;for(let[e,r]of[[n,n],[n,-n],[-n,n],[-n,-n]])t.add(new D(1.95,3.2,6),16777215,K(e,10.05,r));return t.add(new D(2.8,4.6,8),16777215,K(0,14.3,0,Math.PI/8)),t}function Yt(){let e=new G;return e.add(new N(5.2,.4,4.2),H.stoneDark,K(0,.2,0)),e.add(new N(4.6,2.6,3.6),H.stone,K(0,1.7,0)),e.add(new N(.2,1.6,1.3),4930350,K(2.32,1.2,0)),e.add(new O(.08,.1,5.2,5),H.woodDark,K(-1.6,3,2.2)),e}function Xt(){let e=new G;return e.add(new D(2,1.8,4),16777215,K(0,3.9,0,Math.PI/4,1.85,1,1.4)),e}var Zt=class{heightAt;group=new d;hqs=[];barracks=[[],[]];selection;selected=null;materials=[];labels=[];stoneMaterial;bodyGeo;roofGeo;initialised=!1;time=0;fire=1;defenseOn=!0;emitAcc=0;flames;flameMatrix=new E;flamePos=new A;flameScale=new A;flameQuat=new se;crackMaterial=new s({color:3024416,side:2,polygonOffset:!0,polygonOffsetFactor:-2,polygonOffsetUnits:-2});onChange=null;brazierEmitter=null;constructor(e,t,n){this.heightAt=n,this.group.name=`structures`;let r=ot();this.materials.push(this.crackMaterial),this.materials.push(r);let i=qt(t).build(),a=Jt(t).build();this.stoneMaterial=r,this.bodyGeo=Yt().build(),this.roofGeo=Xt().build();for(let n of[0,1]){let o=e.hq[n],s=new d,c=new d,l=new v(i,r),u=new M({color:R[n],flatShading:!0,vertexColors:!0});this.materials.push(u);let f=new v(a,u),p=new M({color:B[n],side:2,flatShading:!0});this.materials.push(p);let m=new v(new O(.08,.08,4,5),r);m.position.set(0,18,0);let h=new v(new N(.05,1.6,2.6),p);h.position.set(0,19.1,1.35);for(let e of[l,f,m,h])e.castShadow=!0,e.receiveShadow=!0;c.add(l,f,m,h);let g=this.heightAt(o.x,o.y),_=Ut(t).map(([e,t])=>{let n=o.x+e,r=o.y+t;return{x:n,y:r,h:g+Ht-this.heightAt(n,r),top:g}});s.add(c),s.position.set(o.x,g,o.y),s.rotation.y=n===0?0:Math.PI,this.group.add(s);let y=Tt(t,11+n*17),b=new v(y.geometry,this.crackMaterial);b.renderOrder=1,b.visible=!1,b.name=`hq-cracks`,c.add(b),this.hqs.push({group:s,body:c,flag:h,flagMaterial:p,braziers:_,cracks:y,cracksDrawn:0,crackMesh:b,shakeUntil:0,fall:0});for(let t of e.barracks[n])this.addBarracks(n,t.x,t.y,1)}let o=new s({vertexColors:!0});this.materials.push(o),this.flames=new T(Wt(),o,2*Ut(1).length),this.flames.frustumCulled=!1,this.group.add(this.flames);let c=new s({color:B[0],transparent:!0,opacity:.85,depthWrite:!1,side:2});this.materials.push(c),this.selection=new v(new fe(4.2,4.9,40).rotateX(-Math.PI/2),c),this.selection.visible=!1,this.selection.renderOrder=3,this.group.add(this.selection)}addBarracks(e,t,n,r){let i=new d,a=new v(this.bodyGeo,this.stoneMaterial),o=new M({color:R[e],flatShading:!0,vertexColors:!0});this.materials.push(o);let s=new v(this.roofGeo,o);for(let e of[a,s])e.castShadow=!0,e.receiveShadow=!0;let c=new dt(128,160);this.labels.push(c);let l=new ce({map:c.texture,transparent:!0,depthWrite:!1});this.materials.push(l);let u=new de(l);u.center.set(.5,0),u.position.set(-1.6,3.4,2.2),u.scale.set(2.2,2.75,1),i.add(a,s,u),i.position.set(t,this.heightAt(t,n),n),i.rotation.y=e===0?0:Math.PI,i.userData.barracks={side:e,index:this.barracks[e].length},this.group.add(i),this.barracks[e].push({group:i,roof:s,roofMaterial:o,banner:u,label:c,shakeUntil:0,appear:r}),this.applyAppear(this.barracks[e][this.barracks[e].length-1])}applyAppear(e){let t=Kt(e.appear);e.group.scale.set(.6+.4*Math.min(1,t),Math.max(.01,t),.6+.4*Math.min(1,t)),e.banner.visible=e.appear>.55}pickTargets(){return this.barracks.flatMap(e=>e.map(e=>e.group))}setSelected(e,t){this.selected=t===null?null:{side:e,index:t},this.selected&&this.selection.material.color.setHex(B[e])}hitHq(e){let t=this.hqs[e];t&&(t.shakeUntil=this.time+.15)}update(e,t,n){this.time=t;let r=this.initialised;this.initialised=!0;for(let t of[0,1]){let n=e.barracks[t];for(let e=this.barracks[t].length;e<n.length;e++){let i=n[e];this.addBarracks(t,i.x,i.y,+!r),this.onChange?.()}}let i=!e.suddenDeath;if(i!==this.defenseOn&&(this.defenseOn=i,!i))for(let e of this.hqs)for(let t of e.braziers){for(let e=0;e<6;e++)this.brazierEmitter?.(t.x,t.y,t.h,!0);this.brazierEmitter?.(t.x,t.y,t.h,!1)}r||(this.fire=+!!i),this.fire+=(+!!i-this.fire)*(1-Math.exp(-n*2.5));for(let r of e.hqs)this.updateHq(r,t,n);for(this.emitAcc+=n*(i?2.2:3);this.emitAcc>=1;){--this.emitAcc;for(let e of this.hqs){if(e.fall>.2)continue;let t=e.braziers[Math.floor(Math.random()*e.braziers.length)];this.brazierEmitter?.(t.x,t.y,t.h,i)}}for(let r of[0,1]){for(let n of e.barracks[r])this.updateBarracks(n,t);for(let e of this.barracks[r])e.appear>=1||(e.appear=Math.min(1,e.appear+n/.9),this.applyAppear(e),e.appear>=1&&this.onChange?.())}if(this.selected){let e=this.barracks[this.selected.side]?.[this.selected.index];if(this.selection.visible=!!e,e){this.selection.position.set(e.group.position.x,e.group.position.y+.15,e.group.position.z);let n=1+.06*Math.sin(t*5);this.selection.scale.set(n,1,n)}}else this.selection.visible=!1}updateHq(e,t,n){let r=this.hqs[e.side];if(!r)return;let i=t<r.shakeUntil?.25:e.crumbling&&!e.destroyed?.05:0;r.body.position.x=i>0?(Math.random()-.5)*i:0,r.body.position.z=i>0?(Math.random()-.5)*i:0;let a=e.maxHp>0?e.hp/e.maxHp:1,o=Ct(e.decayed,e.maxHp),s=o>0?Pt(r.cracks.thresholds,o):0;if(s!==r.cracksDrawn&&(r.cracksDrawn=s,r.cracks.geometry.setDrawRange(0,r.cracks.ends[s-1]??0),r.crackMesh.visible=s>0),r.body.rotation.z=(1-a)*.05,e.collapsed){r.fall=Math.min(1,r.fall+n*.85);let e=r.fall*r.fall*(3-2*r.fall);r.body.scale.set(1+.1*e,1-.55*e,1+.1*e),r.body.position.y=-2.4*e,r.body.rotation.x=.09*e,r.body.rotation.z+=.07*e}else e.destroyed&&(r.fall=Math.min(1,r.fall+n*.8)),r.body.scale.y=1-.7*r.fall*r.fall;r.flag.visible=r.fall<.3;let c=1-this.fire;r.flag.position.y=19.1-1.3*c,r.flagMaterial.color.setHex(B[e.side]).lerp(Gt,c*.55);let l=this.fire>.02&&r.fall<.3;for(let n=0;n<r.braziers.length;n++){let i=r.braziers[n],a=.82+.18*Math.sin(t*(13+n)+n*1.7+e.side*2.3),o=l?this.fire*a:0;this.flamePos.set(i.x,i.top+Ht+.22,i.y),this.flameScale.set(o,o*(.9+.2*Math.sin(t*9+n)),o),this.flameMatrix.compose(this.flamePos,this.flameQuat,this.flameScale),this.flames.setMatrixAt(e.side*r.braziers.length+n,this.flameMatrix)}this.flames.instanceMatrix.needsUpdate=!0,r.flag.rotation.y=Math.sin(t*2.1+e.side)*.25}updateBarracks(e,t){let n=this.barracks[e.side]?.[e.index];if(!n)return;n.roofMaterial.color.setHex(e.neutralized?H.roofNeutral:R[e.side]),n.label.update(`${e.weapon}|${e.level}|${+!!e.neutralized}`,t=>ht(t,n.label.width,n.label.height,e.weapon,e.level,R[e.side],e.neutralized)),n.banner.material.rotation=Math.sin(t*1.7+e.index+e.side*3)*.04;let r=t<n.shakeUntil;n.group.children[0].position.x=r?(Math.random()-.5)*.15:0}updateScale(e){let t=Math.max(1,e/90);for(let e of this.barracks)for(let n of e)n.banner.scale.set(1.8*t,2.25*t,1)}dispose(){this.group.traverse(e=>{e instanceof v&&e.geometry.dispose()});for(let e of this.materials)e.dispose();for(let e of this.labels)e.dispose()}},Qt=2,$t=256,en=96,tn=.1;function nn(e,t,n,r=8){let i=t*n*r;return!(e>0)||!(i>0)?0:1-Math.exp(-e/i)}var rn=`
uniform vec3 uColor;
uniform vec3 uDeep;
uniform vec3 uRim;
uniform float uTime;
uniform float uOpacity;
uniform float uHeal;
uniform float uReserve;
uniform float uValid;
in vec2 vP;
void main() {
  float r = length(vP);
  if (r > 1.0) discard;
  float a01 = fract(atan(vP.x, vP.y) / 6.2831853 + 1.0);
  float band = smoothstep(0.92, 0.95, r) * (1.0 - smoothstep(0.985, 1.0, r));
  float inner = 1.0 - smoothstep(0.88, 0.93, r);
  vec3 c;
  float a;
  if (uValid > 0.5) {
    // Rim gauge: solid arc for the reserve left, faint dashes for what is used.
    float onArc = step(a01, uReserve);
    float dash = step(0.5, fract(a01 * 56.0));
    // Healing: ripples drawn towards the centre and a slow breath.
    float wave = fract(r * 2.4 + uTime * 0.75);
    float ripple = smoothstep(0.0, 0.1, wave) * (1.0 - smoothstep(0.1, 0.26, wave)) * inner;
    float breathe = 0.5 + 0.5 * sin(uTime * 2.6);
    float glow = (1.0 - r * 0.6) * inner;
    a = inner * (0.1 + 0.05 * smoothstep(0.5, 0.9, r))
      + band * mix(0.22 * dash, 0.9, onArc)
      + uHeal * (0.42 * ripple + (0.12 + 0.1 * breathe) * glow);
    c = mix(uColor, uDeep, band * onArc * 0.75);
    c = mix(c, vec3(1.0), uHeal * ripple * 0.35);
    // Thin bright line inside the rim, in the side's light colour.
    float line = smoothstep(0.895, 0.905, r) * (1.0 - smoothstep(0.91, 0.92, r));
    c = mix(c, uRim, line);
    a += line * 0.55;
  } else {
    float dash = step(0.45, fract(a01 * 24.0));
    float hatch = step(0.62, fract((vP.x + vP.y) * 5.0)) * inner;
    a = band * dash * 0.85 + hatch * 0.22 + inner * 0.06;
    c = uColor;
  }
  gl_FragColor = vec4(c, min(0.92, a) * uOpacity);
  #include <colorspace_fragment>
}
`;function an(){let e=new G,t=U.canvas;e.add(new N(3.9,.04,3.1),14274228,K(0,.02,0));let n=1.5;e.add(new O(n,n,3.4,3,1,!1,Math.PI/2),t,K(0,n/2,0,0,1,1,1,0,Math.PI/2)),e.add(new O(.8,.8,.06,3,1,!1,Math.PI/2),5982778,K(1.71,.4,0,0,1,1,1,0,Math.PI/2));for(let t of[-.62,.62])e.add(new O(.09,.09,1.4,5),14998468,K(1.74,.75,t,0,1,1,1,t>0?.5:-.5));for(let t of[-1.85,1.85])e.add(new N(.08,2.5,.08),H.woodDark,K(t,1.25,0));for(let[t,n,r,i]of[[-2.25,0,.55,0],[2.25,0,-.55,0],[0,1.75,0,-.75],[0,-1.75,0,.75]])e.add(new N(.025,1.9,.025),H.rope,K(t,.75,n,0,1,1,1,i,r));e.add(new N(1.7,.08,.6),t,K(.1,.42,1.95));for(let t of[-.65,.85])for(let n of[1.72,2.18])e.add(new N(.06,.4,.06),H.wood,K(t,.2,n));return e.add(new N(.7,.55,.6),H.wood,K(-2.15,.28,-.9,.2)),e.add(new N(.5,.42,.5),H.plank,K(-2.2,.76,-.85,-.3)),e.add(new O(.3,.27,.7,7),H.woodDark,K(-2,.35,.7)),e.add(new N(.07,1.6,.07),H.woodDark,K(2.05,.8,1.05)),e.add(new N(.45,.05,.05),H.woodDark,K(2.05,1.58,.88)),e.add(new O(.05,.06,3.9,5),H.woodDark,K(-1.85,1.95,0)),e.build()}function on(){let e=new G,t=1.5,n=Math.atan2(t*.866,t*1.5);for(let r of[1,-1]){let i=r*(t*.866-.42*Math.tan(n))+r*.03;e.add(new N(3.44,.62,.04),16777215,K(0,.44999999999999996,i,0,1,1,1,-r*n));let a=1.55,o=r*(t*.866-a*Math.tan(n))+r*.03;e.add(new N(3.44,.22,.04),16777215,K(0,a,o,0,1,1,1,-r*n))}return e.add(new N(3.6,.14,.26),16777215,K(0,2.27,0)),e.add(new N(.05,.62,1.1),16777215,K(-1.85,3.5,.58)),e.build()}var sn=class{heightAt;nearestRoute;healPerSecond;group=new d;items=new Map;neutralGeometry=an();colourGeometry=on();lanternGeometry=new N(.26,.34,.26);neutralMaterial=new M({vertexColors:!0,flatShading:!0});sideMaterials=[0,1].map(e=>new M({color:R[e],vertexColors:!0,flatShading:!0}));ghostMaterial=new M({vertexColors:!0,flatShading:!0,transparent:!0,opacity:.55,depthWrite:!1,emissive:new j(0)});ghost=null;ghostKey=``;initialised=!1;time=0;onChange=null;constructor(e,t,n){this.heightAt=e,this.nearestRoute=t,this.healPerSecond=n,this.group.name=`infirmaries`}build(e,t,n,r,i,a,o){let c=new d,l=new d,u=new v(this.neutralGeometry,o?this.ghostMaterial:this.neutralMaterial),f=new v(this.colourGeometry,o?this.ghostMaterial:this.sideMaterials[t]);o||(u.castShadow=f.castShadow=!0,u.receiveShadow=f.receiveShadow=!0);let p=new s({color:U.emberLight}),m=new v(this.lanternGeometry,p);m.position.set(2.05,1.36,.72),m.visible=!o,l.add(u,f,m);let h=a/Qt;l.scale.setScalar(h),c.add(l);let g=nt(),_=new P({vertexShader:it,fragmentShader:rn,transparent:!0,depthWrite:!1,uniforms:{uColor:{value:new j(U.heal)},uDeep:{value:new j(U.healDeep)},uRim:{value:new j(B[t])},uTime:{value:0},uOpacity:{value:0},uHeal:{value:0},uReserve:{value:1},uValid:{value:1}}}),y=new v(g,_);y.renderOrder=2,rt(g,n,r,i,this.heightAt,tn);let b=new dt($t,en),x=new de(new ce({map:b.texture,transparent:!0,depthWrite:!1,depthTest:!1}));x.center.set(.5,0),x.renderOrder=10,c.add(x),this.group.add(c,y);let S={id:e,side:t,group:c,body:l,halo:y,haloMaterial:_,haloGeometry:g,lantern:m,lanternMaterial:p,plaque:x,label:b,appear:0,leaving:0,ending:null,glow:0,reserve:1,shakeUntil:0,seen:!0,scale:h};return this.place(S,n,r),S}place(e,t,n){e.group.position.set(t,this.heightAt(t,n),n);let r=this.nearestRoute(t,n);e.group.rotation.y=r?Math.atan2(-(r.y-n),r.x-t):0}hit(e){let t=this.items.get(e);t&&(t.shakeUntil=this.time+.14)}end(e,t){let n=this.items.get(e);n&&!n.ending&&(n.ending=t)}update(e,t,n,r,i){this.time=n;let a=this.initialised;this.initialised=!0;for(let e of this.items.values())e.seen=!1;for(let n of e){let e=this.items.get(n.id);e||(e=this.build(n.id,n.side,n.x,n.y,n.radius,n.bodyRadius,!1),a||(e.appear=1),this.items.set(n.id,e),this.onChange?.()),e.seen=!0;let i=nn(n.healed,this.healPerSecond,t);e.glow+=(i-e.glow)*(1-Math.exp(-r/(i>e.glow?.18:.45))),e.reserve=n.maxReserve>0?n.reserve/n.maxReserve:0;let o=n.maxHp>0?n.hp/n.maxHp:null;e.label.update(`${Math.ceil(n.reserve)}|${o===null?`-`:Math.round(o*40)}`,e=>bt(e,$t,en,n.reserve,n.maxReserve,o,R[n.side],z[n.side],U.heal))}for(let[e,t]of this.items)!t.seen&&(t.ending||=`depleted`,t.leaving+=r/(t.ending===`destroyed`?.35:1.4),t.leaving>=1)?(this.dispose1(t),this.items.delete(e),this.onChange?.()):this.animate(t,n,r,i);this.ghost?.group.visible&&this.animate(this.ghost,n,r,i)}animate(e,t,n,r){e.appear=Math.min(1,e.appear+n/.7);let i=Kt(e.appear),a=e.leaving,o=Math.max(.01,i),s=.7+.3*Math.min(1,i),c=Math.min(1,e.appear*2.5);e.ending===`destroyed`?(o*=Math.max(.01,1-a*a),s*=1+.25*a,c*=1-a):e.ending===`depleted`&&a>0&&(o*=Math.max(.01,1-.85*a*a),c*=1-Math.min(1,a*1.6)),e.body.scale.set(e.scale*s,e.scale*o,e.scale*s);let l=t<e.shakeUntil;e.body.position.x=l?(Math.random()-.5)*.25:0,e.body.position.z=l?(Math.random()-.5)*.25:0;let u=e.ending?0:e.glow,d=e.haloMaterial.uniforms;d.uTime.value=t,d.uOpacity.value=c*(e===this.ghost?.9:1),d.uHeal.value=u,d.uReserve.value=e.reserve;let f=e.ending===`depleted`?Math.max(0,1-a*3):1,p=.85+.15*Math.sin(t*11+e.id*3.1);e.lanternMaterial.color.setHex(U.emberLight).lerp(cn,u).multiplyScalar((.35+.65*Math.max(u,.45))*p*f+.12);let m=Math.max(1,r/70);e.plaque.scale.set(3.3*m,1.24*m,1),e.plaque.position.y=(3.4+Math.max(0,r-70)*.02)*e.scale,e.plaque.visible=!e.ending&&e.appear>.4}setPreview(e,t,n){if(!e){this.ghost&&(this.ghost.group.visible=!1,this.ghost.halo.visible=!1);return}let r=`${e.side}|${e.radius.toFixed(2)}|${+!!e.valid}`;if(!this.ghost||r!==this.ghostKey){this.ghost&&this.dispose1(this.ghost),this.ghost=this.build(-1,e.side,e.x,e.y,e.radius,t,!0),this.ghost.appear=1,this.ghostKey=r;let i=this.ghost.haloMaterial.uniforms;i.uValid.value=+!!e.valid,e.valid||i.uColor.value.setHex(U.forbidden),this.ghostMaterial.color.setHex(e.valid?B[e.side]:10132122),this.ghostMaterial.emissive.setHex(e.valid?0:U.forbidden),this.ghostMaterial.emissiveIntensity=e.valid?0:.35,this.ghost.label.update(`ghost`,t=>bt(t,$t,en,n,n,null,R[e.side],z[e.side],U.heal,e.valid?`valid`:`forbidden`))}let i=this.ghost;i.group.visible=!0,i.halo.visible=!0,this.place(i,e.x,e.y),rt(i.haloGeometry,e.x,e.y,e.radius,this.heightAt,.12000000000000001),i.glow=e.valid?.35:0}dispose1(e){this.group.remove(e.group,e.halo),e.haloGeometry.dispose(),e.haloMaterial.dispose(),e.lanternMaterial.dispose(),e.plaque.material.dispose(),e.label.dispose()}dispose(){for(let e of this.items.values())this.dispose1(e);this.items.clear(),this.ghost&&this.dispose1(this.ghost),this.neutralGeometry.dispose(),this.colourGeometry.dispose(),this.lanternGeometry.dispose(),this.neutralMaterial.dispose();for(let e of this.sideMaterials)e.dispose();this.ghostMaterial.dispose()}},cn=new j(15335410),J={confettiPerFrame:1800,dustPerFrame:14,arrowsPerFrame:160,sparklesPerGatePerFrame:3,arrivalsPerSitePerFrame:1,sparksPerFrame:700,goldenPassesPerFrame:4,chargeDustPerSnapshot:22,chargeStreaksPerSnapshot:36,healMotesPerSnapshot:5,impactsPerFrame:16},ln=I(`cavalry`),Y=e=>{let t=new j(e);return[t.r,t.g,t.b]},un=class{sys;hooks;sideColors=R.map(Y);sideLight=B.map(Y);sideDark=z.map(Y);gold=Y(V.gold);white=Y(16777215);dust=Y(H.sand);smoke=Y(15921126);stone=Y(H.stone);stoneDark=Y(H.rockDark);crater=Y(7035460);wood=Y(13214330);plank=Y(H.plank);straw=Y(V.straw);horse=Y(V.horse);clods=[Y(7035460),Y(H.sand),Y(H.grassDark),Y(9074006)];shells=[];time=0;confettiThisFrame=0;dustThisFrame=0;arrowsThisFrame=0;sparksThisFrame=0;impactsThisFrame=0;goldenPasses=0;heal=Y(U.heal);healWhite=Y(15400947);goldBright=Y(U.gold);goldLight=Y(U.goldLight);goldDeep=Y(U.goldDeep);ember=Y(U.ember);emberLight=Y(U.emberLight);canvasC=Y(U.canvas);soot=Y(6182480);volleyWood=R.map(e=>{let t=new j(9067059).lerp(new j(e),.4);return[t.r,t.g,t.b]});volleyTiming={waves:1,waveInterval:0};stoneOnly=[Y(H.stoneDark)];stoneDarkOnly=[Y(H.earth)];hqRadius=6;crumbleAcc=[0,0];collapses=[];sparkles=new Map;arrivals=new Map;stats={confetti:0,puffs:0,arrows:0,shells:0};constructor(e,t){this.sys=e,this.hooks=t}beginFrame(e){this.time=e,this.confettiThisFrame=0,this.dustThisFrame=0,this.arrowsThisFrame=0,this.sparkles.clear(),this.sparksThisFrame=0,this.impactsThisFrame=0,this.goldenPasses=0,this.arrivals.clear()}record(e,t,n,r,i){let a=e.ring.alloc(),o=e.ring.data;o[a]=t[0],o[a+1]=t[1],o[a+2]=t[2],o[a+3]=t[3],o[a+4]=n[0],o[a+5]=n[1],o[a+6]=n[2],o[a+7]=n[3],o[a+8]=r[0],o[a+9]=r[1],o[a+10]=r[2],o[a+11]=r[3],o[a+12]=i[0],o[a+13]=i[1],o[a+14]=i[2],o[a+15]=i[3]}confetti(e,t,n,r,i,a=3.2,o=1.6,s=.17,c=0){let l=Math.min(r,J.confettiPerFrame-this.confettiThisFrame);if(l<=0)return;this.confettiThisFrame+=l,this.stats.confetti+=l;let u=this.hooks.heightAt(e,t);for(let r=0;r<l;r++){let l=Math.random()*Math.PI*2,d=o*(.3+Math.random()*.7),f=i[r%i.length];this.record(this.sys.confetti,[e+(Math.random()-.5)*.4,u+n,t+(Math.random()-.5)*.4,this.time+c],[Math.cos(l)*d,a*(.6+Math.random()*.6),Math.sin(l)*d,2.4+Math.random()*1.6],[f[0],f[1],f[2],s*(.75+Math.random()*.5)],[Math.random(),2.4,u,9.8])}}puff(e,t,n,r,i,a,o,s=.6,c=.6,l=.9){let u=this.hooks.heightAt(e,t);this.stats.puffs++,this.record(this.sys.puffs,[e,u+n,t,this.time],[(Math.random()-.5)*c,s,(Math.random()-.5)*c,o],[r[0],r[1],r[2],i],[a,l,1.2,0])}ring(e,t,n,r,i,a,o,s,c,l=1,u=.4,d=.06){let f=this.hooks.heightAt(t,n);this.record(e,[t,f+d,n,this.time],[r,i,a,0],[o[0],o[1],o[2],s],[c,l,u,0])}arrow(e,t,n,r,i){if(this.arrowsThisFrame++>=J.arrowsPerFrame)return;this.stats.arrows++;let a=Math.hypot(r-t,i-n),o=Math.max(.18,a/26),s=this.hooks.heightAt(t,n)+1.25,c=this.hooks.heightAt(r,i)+.55,l=this.wood,u=(Math.random()-.5)*.8,d=(Math.random()-.5)*.8;this.record(this.sys.arrows,[t,s,n,this.time],[r+u,c,i+d,o],[l[0],l[1],l[2],.6+a*.11],[.7,1,0,0])}spark(e,t,n,r,i,a,o,s,c,l=0,u=.5,d=0,f=0){if(this.sparksThisFrame>=J.sparksPerFrame)return;this.sparksThisFrame++;let p=this.hooks.heightAt(e,t),m=this.sys.sparks,h=m.ring.alloc(),g=m.ring.data;g[h]=e,g[h+1]=p+n,g[h+2]=t,g[h+3]=this.time+f,g[h+4]=r,g[h+5]=i,g[h+6]=a,g[h+7]=o,g[h+8]=s[0],g[h+9]=s[1],g[h+10]=s[2],g[h+11]=c,g[h+12]=l,g[h+13]=u,g[h+14]=Math.random(),g[h+15]=d}goldSparkle(e,t,n,r){let i=Math.random()*Math.PI*2,a=r*Math.sqrt(Math.random()),o=Math.random()<.5?this.goldLight:this.goldBright;this.spark(e+Math.cos(i)*a,t+Math.sin(i)*a*.5,n,0,.5+Math.random()*.6,0,1.6+Math.random()*1.2,o,.14+Math.random()*.12,-.15,.4,.35)}emberAt(e,t,n,r){let i=Math.random()<.6?this.ember:this.emberLight;this.spark(e,t,n,(Math.random()-.5)*.6,.7+Math.random()*1.1,(Math.random()-.5)*.6,3.5+Math.random()*2.5,i,(.1+Math.random()*.1)*r,-.08,.3,.6*r)}brazier(e,t,n,r){r?this.spark(e+(Math.random()-.5)*.4,t+(Math.random()-.5)*.4,n+.4,(Math.random()-.5)*.4,1.4+Math.random()*1.2,(Math.random()-.5)*.4,.7+Math.random()*.6,Math.random()<.5?this.ember:this.emberLight,.12+Math.random()*.08,-.4,.8,.15):this.puff(e,t,n+.3,this.soot,1.5,.6,3.6,1.6,.5,2.4)}volley(e,t,n,r,i,a,o){let{waves:s,waveInterval:c}=this.volleyTiming,l=Math.round(Math.min(110,Math.max(36,a*a*1.1))),u=this.volleyWood[e]??this.wood,d=Math.hypot(r-t,i-n),f=this.sys.arrows;for(let e=0;e<s;e++)for(let s=0;s<l;s++){let s=Math.random()*Math.PI*2,l=a*Math.sqrt(Math.random()),p=r+Math.cos(s)*l,m=i+Math.sin(s)*l,h=t+(Math.random()-.5)*9,g=n+(Math.random()-.5)*9,_=Math.max(.3,o*(.9+Math.random()*.1)),v=f.ring.alloc(),y=f.ring.data;y[v]=h,y[v+1]=this.hooks.heightAt(h,g)+1.6,y[v+2]=g,y[v+3]=this.time+e*c+(o-_),y[v+4]=p,y[v+5]=this.hooks.heightAt(p,m)+.3,y[v+6]=m,y[v+7]=_,y[v+8]=u[0],y[v+9]=u[1],y[v+10]=u[2],y[v+11]=4+d*.32+Math.random()*2,y[v+12]=2.2+Math.random(),y[v+13]=1.5,y[v+14]=0,y[v+15]=0,this.stats.arrows++}}chargeTrails(e){if(e.charges.length===0)return;let t=e.soldiers,n=0,r=0;for(let i of e.charges){let e=i.duration>0?i.timeLeft/i.duration:0,a=.05+.1*Math.min(1,e*3);for(let e=0;e<t.count;e++){if(t.side[e]!==i.side||t.route[e]!==i.route||t.weapon[e]===L)continue;let o=t.x[e]-t.prevX[e],s=t.y[e]-t.prevY[e];if(o*o+s*s<1e-4||Math.random()>a)continue;let c=t.x[e]-o*6,l=t.y[e]-s*6;if(n<J.chargeDustPerSnapshot&&Math.random()<.45&&(n++,this.puff(c,l,.25,this.dust,.8+Math.random()*.5,.34,1.1,.4,.8,1.5)),r<J.chargeStreaksPerSnapshot){r++;let e=this.sideLight[i.side];this.spark(c,l,.5+Math.random()*.8,-o*4,.2,-s*4,.5,e,.16,0,2.5)}}}}healMotes(e,t,n){let r=t.soldiers,i=r.count;for(let a of e){let e=nn(a.healed,n,t.dt);if(e<=.01)continue;let o=Math.floor(e*J.healMotesPerSnapshot+Math.random());if(o<=0)continue;let s=a.radius*a.radius,c=0,l=i>0?Math.floor(Math.random()*i):0;for(let e=0;e<i&&c<o;e++){let t=(l+e)%i;if(r.side[t]!==a.side||r.hp[t]>=.999)continue;let n=r.x[t]-a.x,o=r.y[t]-a.y;n*n+o*o>s||(c++,this.healMote(r.x[t],r.y[t],.9))}for(let e=c;e<o;e++){let e=Math.random()*Math.PI*2,t=a.radius*.8*Math.sqrt(Math.random());this.healMote(a.x+Math.cos(e)*t,a.y+Math.sin(e)*t,.3)}}}healMote(e,t,n){let r=Math.random()<.65?this.heal:this.healWhite;this.spark(e+(Math.random()-.5)*.5,t+(Math.random()-.5)*.5,n,0,.9+Math.random()*.6,0,1.3+Math.random()*.6,r,.17+Math.random()*.08,-.2,.6,.18)}impact(e,t,n,r){let i=Math.hypot(n-e,r-t),a=i>.001?(n-e)/i:0,o=i>.001?(r-t)/i:0,s=e+(n-e)*.6,c=t+(r-t)*.6;this.ring(this.sys.shocks,s,c,.3,1.9,.34,this.dust,.7,.3,1,.2),this.ring(this.sys.shocks,s,c,.1,1,.14,this.white,.65,.45,1,.1),this.puff(s,c,.6,this.dust,1.9+Math.random()*.6,.62,1.4,.9,1.6,1.7),this.puff(s-a*.7,c-o*.7,.25,this.dust,1.3,.5,1.1,.4,1.2,1.5);let l=Math.min(10,J.confettiPerFrame-this.confettiThisFrame);if(l>0){this.confettiThisFrame+=l,this.stats.confetti+=l;let e=this.hooks.heightAt(s,c);for(let t=0;t<l;t++){let n=(Math.random()-.5)*3,r=1.6+Math.random()*2.6,i=this.clods[t%this.clods.length];this.record(this.sys.confetti,[s,e+.5,c,this.time],[a*r-o*n,3.8+Math.random()*2.6,o*r+a*n,1.8+Math.random()],[i[0],i[1],i[2],.14+Math.random()*.09],[Math.random(),1.4,e,12])}}for(let e=0;e<6;e++){let t=Math.random()*Math.PI*2,n=1.8+Math.random()*2.2;this.spark(s,c,1.3+Math.random()*.3,Math.cos(t)*n+a*1.5,1.2+Math.random()*1.6,Math.sin(t)*n+o*1.5,.3+Math.random()*.2,e%2==0?this.white:this.goldLight,.16,6,2)}}handle(e){for(let t of e)switch(t.type){case`death`:{let e=(t.mult>1?16:9)+(t.cause===1?4:0)+(t.weapon===ln?4:0),n=t.side,r=t.weapon===L?[this.sideColors[n],this.plank,this.sideColors[n],this.straw,this.white]:t.weapon===ln?[this.sideColors[n],this.horse,this.sideLight[n],this.white,this.sideDark[n]]:[this.sideColors[n],this.sideLight[n],this.sideColors[n],this.white,this.sideDark[n]];this.confetti(t.x,t.y,1,e,r);break}case`siteStarted`:this.ring(this.sys.shocks,t.x,t.y,1,7,.6,this.sideLight[t.side],.75,.18),this.puff(t.x,t.y,.3,this.dust,2.2,.4,1.4,.6,2,1.4);break;case`engineerArrived`:{this.hooks.siteArrival?.(t.side,t.site);let e=t.side*64+t.site,n=this.arrivals.get(e)??0;if(n>=J.arrivalsPerSitePerFrame)break;this.arrivals.set(e,n+1);let r=t.x+(Math.random()-.5)*4,i=t.y+(Math.random()-.5)*3.4;this.puff(r,i,.4,this.dust,1.1+Math.random()*.5,.42,1.2,.7,1.2,1.4),this.confetti(r,i,1.2,4,[this.plank,this.stone,this.plank,this.gold],2.6,1.3,.13);break}case`barracksBuilt`:{let e=t.side;this.ring(this.sys.shocks,t.x,t.y,1,14,.8,this.sideLight[e],.85,.14),this.ring(this.sys.shocks,t.x,t.y,.5,8,.45,this.gold,.7,.2);let n=[this.sideColors[e],this.sideLight[e],this.gold,this.white];this.confetti(t.x,t.y,4,110,n,5.5,4.2,.22);for(let e=0;e<8;e++){let n=e/8*Math.PI*2;this.puff(t.x+Math.cos(n)*3.2,t.y+Math.sin(n)*2.6,.4,this.dust,2.4+Math.random(),.55,2.2,.9,2.4,1.6)}this.hooks.shake(t.x,t.y,.18);break}case`siteStalled`:this.ring(this.sys.shocks,t.x,t.y,1,6,.5,this.stoneDark,.6,.2);break;case`siteResumed`:this.ring(this.sys.shocks,t.x,t.y,1,6,.5,this.sideLight[t.side],.6,.2);break;case`siteCancelled`:this.puff(t.x,t.y,.4,this.dust,1.8,.35,1.2,.5,1.5,1.3);break;case`hit`:if(this.dustThisFrame<J.dustPerFrame&&Math.random()<.035){this.dustThisFrame++;let e=(t.x+t.fromX)/2,n=(t.y+t.fromY)/2;this.puff(e,n,.3,this.dust,1+Math.random()*.7,.3,1.5,.5,1.2,1.3)}break;case`arrowShot`:this.arrow(t.side,t.x,t.y,t.toX,t.toY);break;case`impact`:this.impactsThisFrame++<J.impactsPerFrame&&this.impact(t.x,t.y,t.toX,t.toY);break;case`cannonFired`:{let e=this.hooks.heightAt(t.x,t.y)+.75,n=this.hooks.heightAt(t.toX,t.toY)+.1,r=Math.hypot(t.toX-t.x,t.toY-t.y),i={fx:t.x,fy:t.y,fz:e,tx:t.toX,ty:t.toY,tz:n,t0:this.time,flight:Math.max(.1,t.flightTime),apex:1.5+r*.16,side:t.side,nextPuff:this.time};this.shells.push(i),this.stats.shells++;let a=Y(V.iron);this.record(this.sys.shells,[i.fx,i.fz,i.fy,i.t0],[i.tx,i.tz,i.ty,i.flight],[a[0]*.6,a[1]*.6,a[2]*.6,i.apex],[0,1,0,0]);for(let e=0;e<2;e++)this.puff(t.x,t.y,.9,this.smoke,.7+e*.35,.6,1.1,.8,1,1.5);this.ring(this.sys.shocks,t.x,t.y,.2,1.4,.16,this.white,.6,.45,1,.2);break}case`cannonImpact`:{let e=t.side,n=Math.max(1.5,t.radius);this.ring(this.sys.decals,t.x,t.y,n*.75,n*.85,9,this.crater,.45,0,6,.55,.04),this.ring(this.sys.shocks,t.x,t.y,.4,n*1.7,.35,this.gold,.55,.16,1,.15),this.ring(this.sys.shocks,t.x,t.y,.2,n*.9,.18,this.white,.6,.5,1,.1);let r=[this.sideColors[e],this.gold,this.white,this.sideLight[e]];this.confetti(t.x,t.y,.3,32,r,4.6,3.4,.19);for(let e=0;e<3;e++){let r=e/3*Math.PI*2;this.puff(t.x+Math.cos(r)*n*.5,t.y+Math.sin(r)*n*.5,.3,this.dust,1.2+Math.random()*.6,.4,1.5,1,2.2,1.5)}this.hooks.shake(t.x,t.y,.35);break}case`gatePlaced`:this.ring(this.sys.shocks,t.x,t.y,1,10,.6,this.sideLight[t.side],.8,.15),this.puff(t.x,t.y,.5,this.smoke,3,.5,1.4,.8,3,1.5);break;case`gatePassed`:{this.hooks.flashGate(t.gateId);let e=this.sparkles.get(t.gateId)??0;if(e<J.sparklesPerGatePerFrame){this.sparkles.set(t.gateId,e+1);let n=t.kind===Se.Halve?[this.sideColors[t.side],this.white]:[this.gold,this.white,this.sideLight[t.side]];this.confetti(t.x,t.y,2.2,3,n,2.2,1.2,.12)}break}case`gateHit`:this.hooks.hitGate(t.gateId),Math.random()<.5&&this.confetti(t.x,t.y,2.5,2,[this.stone,this.stoneDark],2.5,2,.15);break;case`gateDestroyed`:case`gateDepleted`:{let e=t.type===`gateDestroyed`;if(!e&&t.kind===0)break;let n=[this.sideColors[t.side],this.sideLight[t.side],this.gold,this.stone];this.confetti(t.x,t.y,3,e?70:30,n,e?5:3,e?4:2.5,.2);for(let n=0;n<(e?6:3);n++)this.puff(t.x+(Math.random()-.5)*6,t.y+(Math.random()-.5)*6,1,this.dust,2.5,.55,2,1,2,1.5);e&&this.hooks.shake(t.x,t.y,.25);break}case`hqHit`:if(this.hooks.hitHq(t.side),Math.random()<.35){let e=Math.random()*Math.PI*2,n=t.x+Math.cos(e)*5,r=t.y+Math.sin(e)*5;this.confetti(n,r,3+Math.random()*4,3,[this.stone,this.stoneDark,this.sideDark[t.side]],2.5,2.5,.22),Math.random()<.4&&this.puff(n,r,2,this.smoke,2,.45,2.4,1.4,1,1.5)}break;case`hqDestroyed`:{if(t.collapsed){this.collapse(t.side,t.x,t.y);break}let e=[this.sideColors[t.side],this.sideLight[t.side],this.gold,this.stone,this.white];for(let n=0;n<6;n++)this.confetti(t.x+(Math.random()-.5)*8,t.y+(Math.random()-.5)*8,4+Math.random()*4,80,e,6,5,.25,n*.15);for(let e=0;e<10;e++)this.puff(t.x+(Math.random()-.5)*12,t.y+(Math.random()-.5)*12,2,this.dust,4,.7,3.5,1.2,3,1.8);this.ring(this.sys.shocks,t.x,t.y,2,26,1,this.gold,.9,.12),this.hooks.shake(t.x,t.y,1);break}case`barracksNeutralized`:case`barracksRestored`:this.ring(this.sys.shocks,t.x,t.y,1,7,.6,t.type===`barracksRestored`?this.sideLight[t.side]:this.white,.7,.2),this.puff(t.x,t.y,1,this.smoke,3,.5,1.8,1,2,1.5);break;case`infirmaryPlaced`:{let e=t.side;this.ring(this.sys.shocks,t.x,t.y,1,t.radius,.7,this.heal,.8,.1),this.ring(this.sys.shocks,t.x,t.y,.5,6,.45,this.sideLight[e],.75,.2),this.puff(t.x,t.y,.3,this.dust,2.4,.45,1.5,.6,2.2,1.4);let n=[this.canvasC,this.sideColors[e],this.heal];this.confetti(t.x,t.y,1.5,24,n,3,2.2,.16);for(let e=0;e<18;e++){let n=e/18*Math.PI*2,r=t.radius*.9;this.healMote(t.x+Math.cos(n)*r,t.y+Math.sin(n)*r,.2)}break}case`infirmaryHit`:this.hooks.hitInfirmary?.(t.infirmaryId),Math.random()<.5&&this.confetti(t.x,t.y,1.6,2,[this.canvasC,this.wood],2.4,1.8,.15);break;case`infirmaryDestroyed`:{this.hooks.endInfirmary?.(t.infirmaryId,`destroyed`);let e=t.side,n=[this.canvasC,this.sideColors[e],this.wood,this.canvasC,this.sideLight[e]];this.confetti(t.x,t.y,1.6,70,n,4.8,3.6,.21);for(let e=0;e<5;e++)this.puff(t.x+(Math.random()-.5)*4,t.y+(Math.random()-.5)*4,.8,this.dust,2.4,.55,2,1,2,1.5);this.ring(this.sys.shocks,t.x,t.y,.5,9,.4,this.white,.6,.2),this.hooks.shake(t.x,t.y,.22);break}case`infirmaryDepleted`:this.hooks.endInfirmary?.(t.infirmaryId,`depleted`),this.puff(t.x,t.y,1.6,this.smoke,1.8,.4,2.6,.9,.6,1.6);for(let e=0;e<10;e++)this.healMote(t.x,t.y,1+Math.random());break;case`goldenGateAnnounced`:this.ring(this.sys.shocks,t.x,t.y,2,22,1.2,this.goldBright,.85,.08);for(let e=0;e<30;e++)this.spark(t.x+(Math.random()-.5)*3,t.y+(Math.random()-.5)*3,.3,(Math.random()-.5)*1.5,4+Math.random()*6,(Math.random()-.5)*1.5,2+Math.random(),this.goldLight,.2,0,.4,.3);break;case`goldenGateOpened`:{this.ring(this.sys.shocks,t.x,t.y,1,20,.9,this.goldBright,.95,.1),this.ring(this.sys.shocks,t.x,t.y,.5,11,.5,this.white,.8,.18);let e=[this.goldBright,this.goldLight,this.goldDeep,this.white];this.confetti(t.x,t.y,6,140,e,6,5,.22);for(let e=0;e<70;e++)this.goldSparkle(t.x,t.y,Math.random()*12,7);this.puff(t.x,t.y,.5,this.goldLight,4,.5,1.6,.8,3,1.6),this.hooks.shake(t.x,t.y,.3);break}case`goldenGateClaimed`:{let e=t.side;this.ring(this.sys.shocks,t.x,t.y,1,16,.8,this.sideLight[e],.9,.12);let n=[this.sideColors[e],this.sideLight[e],this.goldBright,this.white];this.confetti(t.x,t.y,8,70,n,4,4,.22);break}case`goldenGatePassed`:if(this.hooks.flashGolden?.(),this.goldenPasses<J.goldenPassesPerFrame){this.goldenPasses++;let e=[this.goldBright,this.goldLight,this.white];this.confetti(t.x,t.y,2.4,3,e,2.4,1.3,.13);for(let e=0;e<3;e++)this.goldSparkle(t.x,t.y,1+Math.random()*2,1.2)}break;case`goldenGateClosed`:if(this.hooks.closeGolden?.(t.expired),t.expired){for(let e=0;e<6;e++)this.puff(t.x+(Math.random()-.5)*8,t.y+(Math.random()-.5)*8,.6,this.goldLight,2.4,.35,2.2,.7,1.5,1.5);for(let e=0;e<40;e++)this.goldSparkle(t.x,t.y,Math.random()*8,6)}else{let e=[this.goldBright,this.goldLight,this.goldDeep,this.white];this.confetti(t.x,t.y,7,180,e,7,6,.23),this.ring(this.sys.shocks,t.x,t.y,1,24,.9,this.goldBright,.9,.1);for(let e=0;e<60;e++)this.goldSparkle(t.x,t.y,Math.random()*12,7);this.hooks.shake(t.x,t.y,.3)}break;case`rallyUsed`:{let e=t.side;if(t.power===we.Volley){let n=this.hooks.volleySource?.(e,t.x,t.y)??null;n&&this.volley(e,n.x,n.y,t.x,t.y,n.radius,t.duration),this.ring(this.sys.shocks,t.x,t.y,.5,4,.5,this.sideLight[e],.7,.25)}else t.power===we.Charge&&(this.ring(this.sys.shocks,t.x,t.y,2,26,1,this.sideLight[e],.85,.08),this.ring(this.sys.shocks,t.x,t.y,1,14,.6,this.white,.7,.15));break}case`volleyImpact`:{let e=Math.max(1,t.radius),n=3+Math.min(5,Math.floor(t.hits/4));for(let r=0;r<n;r++){let n=Math.random()*Math.PI*2,r=e*Math.sqrt(Math.random());this.puff(t.x+Math.cos(n)*r,t.y+Math.sin(n)*r,.2,this.dust,1.3+Math.random()*.8,.45,1.6,.6,1.6,1.5)}this.ring(this.sys.shocks,t.x,t.y,e*.3,e*1.15,.35,this.white,.55,.12),this.ring(this.sys.decals,t.x,t.y,e*.9,e,6,this.crater,.16,0,6,.6,.04),this.hooks.shake(t.x,t.y,.08);break}case`reinforcementsArrived`:{let e=t.side;this.hooks.reinforcements?.(e,t.x,t.y,t.route),this.ring(this.sys.shocks,t.x,t.y,1,18,.8,this.sideLight[e],.9,.12);let n=[this.sideColors[e],this.sideLight[e],this.goldBright,this.white];this.confetti(t.x,t.y,4,90,n,5,4.5,.21);for(let e=0;e<12;e++){let n=e/12*Math.PI*2;this.puff(t.x+Math.cos(n)*5,t.y+Math.sin(n)*4,.4,this.dust,2.6+Math.random(),.55,2.4,.8,2.6,1.6)}this.hooks.shake(t.x,t.y,.15);break}case`suddenDeathStarted`:this.hooks.suddenDeath?.()}}celebrate(e,t){let n=[this.sideColors[e],this.sideLight[e],this.gold,this.white];for(let e of t)this.confetti(e.x,e.y,12,60,n,1.5,6,.3,Math.random()*.6)}crumble(e,t,n,r){let i=this.hqRadius,a=i*.62;for(this.crumbleAcc[e]=(this.crumbleAcc[e]??0)+r*9;this.crumbleAcc[e]>=1;){--this.crumbleAcc[e];let r=Math.random()<.45,o=Math.random()*Math.PI*2,s=r?Math.sign(Math.cos(o))*a:Math.cos(o)*a,c=r?Math.sign(Math.sin(o))*a:Math.sin(o)*a,l=r?7.8+Math.random()*1.5:5.6+Math.random()*.8,u=Math.random()<.5?this.stoneOnly:this.stoneDarkOnly;if(this.confetti(t+s*1.05,n+c*1.05,l,1,u,.8,1.4,.32+Math.random()*.2),Math.random()<.3){let e=i*(.85+Math.random()*.4);this.puff(t+Math.cos(o)*e,n+Math.sin(o)*e,.4,this.dust,1.6+Math.random()*.8,.42,2.2,.6,1.2,1.6)}}}collapse(e,t,n){let r=this.hqRadius;for(let e=0;e<26;e++){let i=e/26*Math.PI*2,a=r*(.6+Math.random()*.7);this.puff(t+Math.cos(i)*a,n+Math.sin(i)*a,.5+Math.random()*3,this.dust,4+Math.random()*3,.75,4.5,.9,3.2,2)}let i=[this.stone,this.stoneDark,this.stone,this.sideDark[e]];for(let e=0;e<5;e++)this.confetti(t+(Math.random()-.5)*r,n+(Math.random()-.5)*r,3+Math.random()*5,36,i,2.6,4.5,.36,e*.2);this.ring(this.sys.decals,t,n,r*1.3,r*1.6,12,this.crater,.4,0,3,.6,.04),this.ring(this.sys.shocks,t,n,2,r*4,1.6,this.dust,.7,.2),this.hooks.shake(t,n,.6),this.collapses.push({x:t,y:n,t0:this.time,next:this.time})}update(){for(let e=this.collapses.length-1;e>=0;e--){let t=this.collapses[e],n=this.time-t.t0;if(n>2.5){this.collapses.splice(e,1);continue}if(this.time<t.next)continue;t.next=this.time+.12;let r=Math.random()*Math.PI*2,i=this.hqRadius*(.8+Math.random()*.8);this.puff(t.x+Math.cos(r)*i,t.y+Math.sin(r)*i,.4,this.dust,3+Math.random()*2,.6,3.5,.7,2.4,1.8),n<1.6&&this.hooks.shake(t.x,t.y,.12)}for(let e=this.shells.length-1;e>=0;e--){let t=this.shells[e],n=(this.time-t.t0)/t.flight;if(n>=1){this.shells.splice(e,1);continue}if(this.time<t.nextPuff)continue;t.nextPuff=this.time+.03;let r=t.fx+(t.tx-t.fx)*n,i=t.fy+(t.ty-t.fy)*n,a=t.fz+(t.tz-t.fz)*n+4*t.apex*n*(1-n),o=this.hooks.heightAt(r,i);if(this.puff(r,i,a-o,this.smoke,.32,.45,.7,.15,.25,1.6),Math.random()<.6){let e=Math.random()<.5?this.sideColors[t.side]:this.gold;this.confetti(r,i,a-o,1,[e],.5,.4,.11)}}for(let e of this.sys.all)e.flush()}},dn=5.8,fn=256,pn=128,mn=46;function hn(e,t,n){let r=n-t;return r>0?Math.max(0,Math.min(1,(e-t)/r)):1}function gn(e,t){return Math.max(0,Math.ceil(t-e-1e-6))}function _n(e){return`×${e+1}`}var vn=`
uniform vec3 uGold;
uniform vec3 uDeep;
uniform vec3 uLight;
uniform float uTime;
uniform float uProgress;
uniform float uOpen;
uniform float uOpacity;
uniform float uFlash;
in vec2 vP;
void main() {
  float r = length(vP);
  if (r > 1.0) discard;
  float ang = atan(vP.x, vP.y);
  float a01 = fract(ang / 6.2831853 + 1.0);
  float band = smoothstep(0.84, 0.88, r) * (1.0 - smoothstep(0.95, 0.99, r));
  float edge = smoothstep(0.83, 0.85, r) * (1.0 - smoothstep(0.86, 0.88, r))
             + smoothstep(0.95, 0.97, r) * (1.0 - smoothstep(0.98, 1.0, r));
  float drawn = max(uOpen, step(a01, uProgress));
  float head = (1.0 - uOpen) * exp(-abs(a01 - uProgress) * 70.0) * band;
  float ticks = step(0.9, fract(a01 * 12.0 + 0.05)) * smoothstep(0.7, 0.72, r) * (1.0 - smoothstep(0.8, 0.82, r));
  float rays = smoothstep(0.6, 0.95, 0.5 + 0.5 * sin(ang * 14.0 + uTime * 0.7))
             * (1.0 - smoothstep(0.25, 0.8, r)) * smoothstep(0.06, 0.16, r);
  float beat = 0.5 + 0.5 * sin(uTime * mix(3.0, 13.0, uProgress * uProgress));
  float core = 1.0 - smoothstep(0.0, 0.7, r);
  float a = band * (drawn * 0.92 + (1.0 - drawn) * 0.16)
          + edge * drawn * 0.5
          + ticks * (0.45 + 0.4 * drawn)
          + rays * (0.16 + 0.24 * uProgress + 0.06 * uOpen)
          + core * (0.1 + 0.12 * beat * (1.0 - uOpen) + 0.05 * uOpen)
          + head * 0.9
          + uFlash * 0.45 * core;
  vec3 c = mix(uGold, uDeep, edge * drawn);
  c = mix(c, uLight, min(1.0, head + rays * 0.4 + uFlash * 0.6));
  gl_FragColor = vec4(c, min(0.96, a) * uOpacity);
  #include <colorspace_fragment>
}
`,yn=`
out vec2 vUv;
out float vFacing;
void main() {
  vUv = uv;
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  vec3 n = normalize(normalMatrix * normal);
  vFacing = abs(dot(n, normalize(-mv.xyz)));
  gl_Position = projectionMatrix * mv;
}
`,bn=`
uniform vec3 uColor;
uniform float uTime;
uniform float uIntensity;
in vec2 vUv;
in float vFacing;
void main() {
  float y = vUv.y;
  float fade = pow(1.0 - y, 1.6) * smoothstep(0.0, 0.03, y);
  float streak = 0.55 + 0.45 * sin(vUv.x * 6.2831853 * 7.0 + y * 9.0 - uTime * 2.4);
  float rise = 0.65 + 0.35 * smoothstep(0.3, 1.0, fract(y * 2.5 - uTime * 0.55));
  // Soft edges: the column is brighter where it is seen through its thickness.
  float body = 0.35 + 0.65 * (1.0 - vFacing);
  gl_FragColor = vec4(uColor, fade * streak * rise * body * uIntensity);
  #include <colorspace_fragment>
}
`,xn=`
out vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`,Sn=`
uniform vec3 uColor;
uniform float uFlash;
uniform float uTime;
uniform float uOpacity;
in vec2 vUv;
void main() {
  float bands = 0.5 + 0.5 * sin(vUv.x * 34.0 + sin(vUv.y * 5.0 + uTime * 2.2) * 2.5);
  float rise = fract(vUv.y * 2.0 - uTime * 0.6);
  float fade = 1.0 - smoothstep(0.1, 1.0, vUv.y);
  float a = (0.2 + 0.2 * bands * rise) * fade + uFlash * 0.55 * (1.0 - vUv.y * 0.4);
  vec3 c = mix(uColor, vec3(1.0, 0.98, 0.88), 0.25 * bands + uFlash * 0.7);
  gl_FragColor = vec4(c, min(0.9, a) * uOpacity);
  #include <colorspace_fragment>
}
`,Cn=`
out vec2 vL;
void main() {
  vL = position.xz;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`,wn=`
uniform vec3 uColor;
uniform vec3 uTrim;
uniform float uFlash;
uniform float uTime;
uniform float uOpacity;
uniform float uHalfWidth;
uniform float uHalfDepth;
uniform float uDir;
in vec2 vL;
void main() {
  vec2 q = abs(vL) / vec2(uHalfWidth, uHalfDepth);
  float m = max(q.x, q.y);
  float border = smoothstep(0.84, 0.92, m);
  float fade = 1.0 - smoothstep(0.93, 1.0, m);
  float pattern;
  if (abs(uDir) > 0.5) {
    float s = vL.y * uDir - uTime * 2.6 - abs(vL.x) * 0.55;
    float ch = fract(s / 1.8);
    pattern = smoothstep(0.0, 0.08, ch) * (1.0 - smoothstep(0.24, 0.32, ch));
  } else {
    float d = fract((abs(vL.x) * 0.5 + abs(vL.y)) / 1.6 - uTime * 0.4);
    pattern = smoothstep(0.0, 0.1, d) * (1.0 - smoothstep(0.2, 0.32, d));
  }
  float a = (0.62 + pattern * 0.25 + border * 0.3) * fade;
  vec3 c = mix(uColor, vec3(1.0, 0.98, 0.86), pattern * 0.55 + uFlash * 0.4);
  c = mix(c, uTrim, border);
  gl_FragColor = vec4(c, min(0.94, a) * uOpacity);
  #include <colorspace_fragment>
}
`;function Tn(e){let t=new G,n=U.gold,r=U.goldDeep,i=U.goldLight;for(let a of[-e,e])t.add(new N(2.3,.6,2.3),r,K(a,.3,0)),t.add(new N(1.8,.3,1.8),n,K(a,.75,0)),t.add(new O(.42,.56,dn,8),n,K(a,3.8,0)),t.add(new N(1.7,.4,1.7),i,K(a,6.8,0)),t.add(new D(.42,1.3,6),i,K(a,7.65,0));return t.add(new _e(e,.34,6,28,Math.PI),n,K(0,7,0,0,1,En,1)),t.add(new _e(e-.5,.12,5,28,Math.PI),i,K(0,7,0,0,1,En,1)),t.add(new N(.9,1,.9),r,K(0,Dn(e),0)),t.build()}var En=.5;function Dn(e){return 7+e*En}function On(e){return Dn(e)+2.6}function kn(e){let t=new G,n=U.gold,r=U.goldLight,i=U.goldDeep,a=Math.max(2.4,e*.42);t.add(new _e(a,.3,6,40),n,K(0,0,0,0,1,1,1,Math.PI/2)),t.add(new _e(a*.8,.1,4,40),r,K(0,-.25,0,0,1,1,1,Math.PI/2));for(let e=0;e<14;e++){let i=e/14*Math.PI*2;t.add(new D(.28,e%2==0?1.4:.9,4),e%2==0?r:n,K(Math.cos(i)*a,e%2==0?.85:.6,Math.sin(i)*a))}for(let e=0;e<4;e++){let n=e/4*Math.PI*2+Math.PI/4;t.add(new N(a-.9,.12,.12),i,K(Math.cos(n)*a/2,0,Math.sin(n)*a/2,-n))}return t.add(new b(.95,1),r,K(0,.1,0)),t.build()}function An(e){let t=new G;for(let n of[-e,e]){for(let e of[1.7,4.09,6.3])t.add(new O(.7,.7,.22,8),16777215,K(n,e,0));for(let e of[-.95,.95])t.add(new N(.75,2.2,.05),16777215,K(n,5.3999999999999995,e))}return t.build()}function jn(){let e=ut(256,256),t=e.getContext(`2d`),n=t.createRadialGradient(128,128,0,128,128,128);n.addColorStop(0,`rgba(255,240,180,0.9)`),n.addColorStop(.25,`rgba(255,210,90,0.35)`),n.addColorStop(1,`rgba(255,200,80,0)`),t.fillStyle=n,t.fillRect(0,0,256,256);for(let e=0;e<18;e++){let n=e/18*Math.PI*2,r=e%2==0?.075:.04,i=t.createLinearGradient(128,128,128+Math.cos(n)*128,128+Math.sin(n)*128);i.addColorStop(0,`rgba(255,236,160,0.8)`),i.addColorStop(1,`rgba(255,200,80,0)`),t.fillStyle=i,t.beginPath(),t.moveTo(128,128),t.arc(128,128,128,n-r,n+r),t.closePath(),t.fill()}let r=new me(e);return r.colorSpace=he,r}var Mn=class{heightAt;group=new d;current=null;fading=null;raysTex=jn();columnGeometry=new O(1,1,mn,28,1,!0).translate(0,mn/2,0);initialised=!1;white=new j(16777215);tmp=new j;onChange=null;sparkle=null;constructor(e){this.heightAt=e,this.group.name=`golden-gates`}build(e){let t=e.halfWidth,n=new d;n.position.set(e.x,this.heightAt(e.x,e.y),e.y),n.rotation.y=Math.atan2(-e.tx,-e.ty);let r=nt(8,72),i=new P({vertexShader:it,fragmentShader:vn,transparent:!0,depthWrite:!1,uniforms:{uGold:{value:new j(U.gold)},uDeep:{value:new j(U.goldDeep)},uLight:{value:new j(U.goldLight)},uTime:{value:0},uProgress:{value:0},uOpen:{value:0},uOpacity:{value:0},uFlash:{value:0}}}),a=new v(r,i);a.renderOrder=2,rt(r,e.x,e.y,t+1.4,this.heightAt,.14,.8);let o=new P({vertexShader:yn,fragmentShader:bn,transparent:!0,depthWrite:!1,blending:2,side:2,uniforms:{uColor:{value:new j(U.gold)},uTime:{value:0},uIntensity:{value:0}}}),s=new v(this.columnGeometry,o),c=Math.max(1.6,t*.24);s.scale.set(c,1,c),s.renderOrder=8;let l=new d,u=Tn(t),f=An(t),p=new M({vertexColors:!0,flatShading:!0,emissive:new j(U.goldDeep),emissiveIntensity:.35}),m=new M({color:16777215,vertexColors:!0,flatShading:!0}),h=new v(u,p),g=kn(t),_=new v(g,p);_.position.y=On(t),_.castShadow=!0;let y=new v(f,m);h.castShadow=y.castShadow=!0;let b=dn+t*En*.85,x=new F(2*t-1.3,b);x.translate(0,b/2+.6,0);let S=new P({vertexShader:xn,fragmentShader:Sn,transparent:!0,depthWrite:!1,side:2,uniforms:{uColor:{value:new j(U.gold)},uFlash:{value:0},uTime:{value:0},uOpacity:{value:1}}}),C=new v(x,S);C.renderOrder=6;let w=2.4,T=new F(2*t,2*w).rotateX(-Math.PI/2);T.translate(0,.18,0);let E=new P({vertexShader:Cn,fragmentShader:wn,transparent:!0,depthWrite:!1,uniforms:{uColor:{value:new j(U.gold)},uTrim:{value:new j(U.goldLight)},uFlash:{value:0},uTime:{value:0},uOpacity:{value:1},uHalfWidth:{value:t},uHalfDepth:{value:w},uDir:{value:0}}}),ee=new v(T,E);ee.renderOrder=3;let te=new ce({map:this.raysTex,transparent:!0,depthWrite:!1,blending:2,opacity:0}),ne=new de(te);ne.position.set(0,On(t),0),ne.renderOrder=7,l.add(h,_,y,C,ee,ne),l.visible=!1;let re=new dt(fn,pn),ie=new de(new ce({map:re.texture,transparent:!0,depthWrite:!1,depthTest:!1}));return ie.center.set(.5,0),ie.renderOrder=10,n.add(l,s,ie),this.group.add(n,a),{id:e.id,group:n,arch:l,ground:a,groundMaterial:i,groundGeometry:r,column:s,columnMaterial:o,goldMaterial:p,trimMaterial:m,veilMaterial:S,sillMaterial:E,rays:ne,raysMaterial:te,plaque:ie,label:re,geometries:[u,g,f,x,T],crown:_,open:!1,rise:0,appear:0,leaving:0,ending:null,owner:-1,claim:0,flash:0,progress:0,sparkAcc:0,x:e.x,y:e.y,hw:t}}flash(){this.current&&(this.current.flash=1)}close(e){let t=this.current;t&&!t.ending&&(t.ending=e?`expired`:`depleted`)}get active(){return this.current!==null}update(e,t,n,r,i){let a=!this.initialised;this.initialised=!0,this.current&&(!e||e.id!==this.current.id)&&(this.fading&&this.dispose1(this.fading),this.fading=this.current,this.fading.ending||(this.fading.ending=`expired`),this.current=null),e&&!this.current&&(this.current=this.build(e),a&&(this.current.appear=1));let o=this.current;if(o&&e){let s=e.phase===`open`;s&&!o.open&&(o.open=!0,o.arch.visible=!0,o.rise=+!!a,this.onChange?.()),o.progress=s?1:hn(t,e.announcedAt,e.opensAt);let c=e.owner;c!==o.owner&&(o.owner=c,c!==-1&&(o.sillMaterial.uniforms.uDir.value=c===0?1:-1,o.sillMaterial.uniforms.uTrim.value.setHex(R[c]),o.flash=1));let l=_n(e.copies),u=s?null:gn(t,e.opensAt),d=c===-1?null:R[c];o.label.update(`${l}|${u??`-`}|${Math.ceil(e.charges)}|${e.owner}`,t=>xt(t,fn,pn,l,e.charges,e.maxCharges,u,d)),this.animate(o,n,r,i)}let s=this.fading;s&&(s.leaving+=r/(s.ending===`depleted`?.55:1.3),s.leaving>=1?(this.dispose1(s),this.fading=null,this.onChange?.()):this.animate(s,n,r,i))}animate(e,t,n,r){e.appear=Math.min(1,e.appear+n/.6),e.open&&(e.rise=Math.min(1,e.rise+n/.7)),e.flash=Math.max(0,e.flash-n*4.5);let i=+(e.owner>=0);e.claim+=(i-e.claim)*(1-Math.exp(-n*8));let a=e.leaving,o=e.appear*(1-a),s=Kt(e.rise),c=e.ending===`depleted`?1-a:1-a*a;e.arch.scale.set(1,Math.max(.01,s*Math.max(.01,c)),1);let l=e.groundMaterial.uniforms;l.uTime.value=t,l.uProgress.value=e.progress,l.uOpen.value=+!!e.open,l.uOpacity.value=o,l.uFlash.value=e.flash;let u=.5+.5*Math.sin(t*(3+10*e.progress*e.progress)),d=.25+.55*e.progress+.15*u*(1-e.progress),f=e.open?.32+.6*(1-e.rise)+.25*e.flash:d;e.columnMaterial.uniforms.uIntensity.value=f*o,e.columnMaterial.uniforms.uTime.value=t,e.column.scale.y=.2+.8*Math.min(1,e.appear*1.4),e.goldMaterial.emissiveIntensity=.32+.45*e.flash+.06*Math.sin(t*2.3),this.tmp.setHex(e.owner>=0?R[e.owner]:16777215),e.trimMaterial.color.copy(this.white).lerp(this.tmp,e.claim),e.trimMaterial.emissive.setHex(e.owner>=0?B[e.owner]:0),e.trimMaterial.emissiveIntensity=.15*e.claim+.4*e.flash,e.veilMaterial.uniforms.uFlash.value=e.flash,e.veilMaterial.uniforms.uTime.value=t,e.veilMaterial.uniforms.uOpacity.value=o,e.sillMaterial.uniforms.uFlash.value=e.flash,e.sillMaterial.uniforms.uTime.value=t,e.sillMaterial.uniforms.uOpacity.value=o,e.raysMaterial.opacity=(.55+.2*Math.sin(t*1.7)+.4*e.flash)*o*e.rise,e.raysMaterial.rotation=t*.12,e.crown.rotation.y=t*.35,e.crown.position.y=On(e.hw)+.25*Math.sin(t*1.4);let p=(e.hw*1.5+6)*(1+.15*e.flash);e.rays.scale.set(p,p,1);let m=Math.max(1,r/70);e.plaque.scale.set(3.9*m,1.95*m,1);let h=e.open?(On(e.hw)+2.2)*Math.max(.3,s):3.5;if(e.plaque.position.y=h+Math.max(0,r-70)*.03,e.plaque.visible=!e.ending&&e.appear>.3,e.open&&!e.ending&&this.sparkle)for(e.sparkAcc+=n*(14+30*e.flash);e.sparkAcc>=1;)--e.sparkAcc,this.sparkle(e.x,e.y,.5+Math.random()*On(e.hw),e.hw+1)}dispose1(e){this.group.remove(e.group,e.ground);for(let t of e.geometries)t.dispose();e.groundGeometry.dispose(),e.groundMaterial.dispose(),e.columnMaterial.dispose(),e.goldMaterial.dispose(),e.trimMaterial.dispose(),e.veilMaterial.dispose(),e.sillMaterial.dispose(),e.raysMaterial.dispose(),e.plaque.material.dispose(),e.label.dispose()}dispose(){this.current&&this.dispose1(this.current),this.fading&&this.dispose1(this.fading),this.current=this.fading=null,this.raysTex.dispose(),this.columnGeometry.dispose()}};function Nn(e,t,n,r){let i=n<r;return i||!(t>0)?{aim:1,landing:i}:{aim:Math.max(0,Math.min(1,1-e/t)),landing:i}}function Pn(e,t){return t>0?Math.max(0,Math.min(1,e/(t*.2))):0}var Fn=`
in vec2 aRibbon; // across (-1..1), distance along the route (m)
out vec2 vR;
void main() {
  vR = aRibbon;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`,In=`
uniform vec3 uColor;
uniform vec3 uLight;
uniform float uTime;
uniform float uOpacity;
uniform float uDirection;
uniform float uValid;
in vec2 vR;
void main() {
  float across = abs(vR.x);
  float edge = smoothstep(0.8, 0.9, across) * (1.0 - smoothstep(0.96, 1.0, across));
  float a;
  vec3 c;
  if (uValid > 0.5) {
    float s = vR.y * uDirection - uTime * 14.0;
    float chevron = fract((s - across * 3.0) / 6.0);
    float band = smoothstep(0.0, 0.06, chevron) * (1.0 - smoothstep(0.2, 0.34, chevron));
    a = 0.26 + edge * 0.7 + band * 0.5 * (1.0 - across * 0.6);
    c = mix(uColor, uLight, band * 0.7 + edge * 0.5);
  } else {
    float dash = step(0.5, fract(vR.y / 2.4 - uTime * 0.5));
    a = 0.08 + edge * dash * 0.8;
    c = uColor;
  }
  gl_FragColor = vec4(c, min(0.92, a) * uOpacity);
  #include <colorspace_fragment>
}
`,Ln=1.6,Rn=`
uniform vec3 uColor;
uniform vec3 uLight;
uniform float uTime;
uniform float uOpacity;
uniform float uAim;
uniform float uLanding;
uniform float uPulse;
uniform float uValid;
uniform float uPreview;
in vec2 vP;
// The mesh spans DISC_SCALE × the radius so the closing ring can start outside it.
const float DISC_SCALE = ${Ln.toFixed(2)};
void main() {
  vec2 p = vP * DISC_SCALE;
  float r = length(p);
  if (r > DISC_SCALE) discard;
  float a01 = fract(atan(p.x, p.y) / 6.2831853 + 1.0);
  float rim = smoothstep(0.9, 0.94, r) * (1.0 - smoothstep(0.98, 1.0, r));
  float inner = 1.0 - smoothstep(0.88, 0.94, r);
  float cross = (1.0 - smoothstep(0.025, 0.045, abs(p.x))) + (1.0 - smoothstep(0.025, 0.045, abs(p.y)));
  cross *= smoothstep(0.12, 0.18, r) * (1.0 - smoothstep(0.55, 0.62, r));
  float dot0 = 1.0 - smoothstep(0.05, 0.08, r);
  vec3 c = uColor;
  float a;
  if (uValid < 0.5) {
    float dash = step(0.45, fract(a01 * 24.0 - uTime * 0.3));
    float hatch = step(0.62, fract((p.x + p.y) * 5.0)) * inner;
    a = rim * dash * 0.85 + hatch * 0.2 + inner * 0.05;
  } else if (uPreview > 0.5) {
    float dash = step(0.4, fract(a01 * 32.0 - uTime * 0.6));
    a = inner * 0.16 + rim * (0.45 + 0.45 * dash) + cross * 0.6 + dot0 * 0.7;
    c = mix(uColor, uLight, rim * dash * 0.6);
  } else {
    // Closing ring: from 1.6 R onto the rim while the arrows fly.
    float target = mix(1.55, 0.96, uAim);
    float closing = (1.0 - uLanding) * (1.0 - smoothstep(0.0, 0.035, abs(r - target)));
    float throb = uPulse * (1.0 - r * 0.5);
    a = inner * (0.18 + 0.14 * uAim + 0.25 * throb) + rim * 0.85 + cross * 0.55 + dot0 * 0.6
      + closing * 0.9;
    c = mix(uColor, uLight, rim * 0.4 + closing + throb * 0.6);
  }
  gl_FragColor = vec4(c, min(0.92, a) * uOpacity);
  #include <colorspace_fragment>
}
`,zn=4.5,Bn=class{heightAt;waves;group=new d;routes;ribbons;charges=new Map;aimRibbon;aimRibbonMeshes;aimRibbonOpacity=0;aimRibbonRoute=-1;volleys=new Map;aimDisc;aimDiscTarget=0;pennantGeometry;poleGeometry;pennantMaterials;poleMaterial=new M({color:7227948,flatShading:!0});pennants=[];constructor(e,t,n){this.heightAt=t,this.waves=n,this.group.name=`rally`,this.routes=e.routes.map(e=>new o(e.key,e.name,e.points,e.widths)),this.ribbons=this.routes.map(e=>this.ribbon(e)),this.aimRibbon=this.ribbonMaterial(0),this.aimRibbonMeshes=this.ribbons.map(e=>{let t=new v(e,this.aimRibbon);return t.visible=!1,t.renderOrder=3,this.group.add(t),t}),this.aimDisc=this.volleyObject(-1),this.aimDisc.material.uniforms.uPreview.value=1;let r=new G().add(new N(.06,1.3,1.9),16777215,K(0,3.9,.95));r.add(new N(.07,.2,2),16777215,K(0,4.55,1)),this.pennantGeometry=r.build(),this.poleGeometry=new O(.06,.08,4.8,5).translate(0,2.4,0),this.pennantMaterials=[0,1].map(e=>new M({color:R[e],vertexColors:!0,flatShading:!0,side:2}))}ribbon(e){let t=1.5,n=Math.ceil(e.length/t)+1,r=new Float32Array(n*2*3),i=new Float32Array(n*2*2),a=[];for(let o=0;o<n;o++){let n=Math.min(e.length,o*t),s=e.pointAt(n),c=e.tangentAt(n),l=e.widthAt(n)/2;for(let e=0;e<2;e++){let t=e===0?-1:1,a=s.x-c.y*l*t,u=s.y+c.x*l*t,d=(o*2+e)*3;r[d]=a,r[d+1]=this.heightAt(a,u)+.15,r[d+2]=u,i[(o*2+e)*2]=t,i[(o*2+e)*2+1]=n}if(o>0){let e=(o-1)*2;a.push(e,e+1,e+2,e+1,e+3,e+2)}}let o=new re;return o.setAttribute(`position`,new k(r,3)),o.setAttribute(`aRibbon`,new k(i,2)),o.setIndex(a),o.computeBoundingSphere(),o}ribbonMaterial(e){return new P({vertexShader:Fn,fragmentShader:In,transparent:!0,depthWrite:!1,side:2,uniforms:{uColor:{value:new j(R[e])},uLight:{value:new j(B[e])},uTime:{value:0},uOpacity:{value:0},uDirection:{value:e===0?1:-1},uValid:{value:1}}})}volleyObject(e){let t=nt(6,48),n=new P({vertexShader:it,fragmentShader:Rn,transparent:!0,depthWrite:!1,uniforms:{uColor:{value:new j(R[0])},uLight:{value:new j(B[0])},uTime:{value:0},uOpacity:{value:0},uAim:{value:0},uLanding:{value:0},uPulse:{value:0},uValid:{value:1},uPreview:{value:0}}}),r=new v(t,n);return r.renderOrder=3,r.visible=!1,this.group.add(r),{key:e,mesh:r,geometry:t,material:n,seen:!1,opacity:0,wavesLeft:0,pulse:0}}setAim(e){if(this.aimDiscTarget=0,this.aimRibbonRoute=-1,!e)return;if(e.power===`volley`){let t=this.aimDisc.material.uniforms;t.uColor.value.setHex(e.valid?B[e.side]:U.forbidden),t.uLight.value.setHex(16777215),t.uValid.value=+!!e.valid,rt(this.aimDisc.geometry,e.x,e.y,e.radius*Ln,this.heightAt,.2),this.aimDiscTarget=1;return}let t=this.aimRibbon.uniforms;t.uColor.value.setHex(e.valid?R[e.side]:U.forbidden),t.uLight.value.setHex(B[e.side]),t.uDirection.value=e.side===0?1:-1,t.uValid.value=+!!e.valid,this.aimRibbonRoute=e.route}reinforcements(e,t,n,r){let i=this.routes[r],a=new d,o=0,s=1,c=4;if(i){let e=i.project(t,n).distance,r=i.tangentAt(e);o=-r.y,s=r.x,c=i.widthAt(e)/2+1.2}for(let r of[-1,-.33,.33,1]){let i=t+o*c*r,l=n+s*c*r,u=new v(this.poleGeometry,this.poleMaterial),f=new v(this.pennantGeometry,this.pennantMaterials[e]),p=new d;p.add(u,f),p.position.set(i,this.heightAt(i,l),l),p.rotation.y=e===0?-Math.PI/2:Math.PI/2,p.userData.phase=r*2.1,a.add(p)}this.group.add(a),this.pennants.push({group:a,t:0})}update(e,t,n,r){for(let e of this.charges.values())e.target=0;for(let t of e){let e=t.route*2+t.side,n=this.charges.get(e);if(!n){let r=this.ribbonMaterial(t.side),i=new v(this.ribbons[t.route],r);i.renderOrder=3,this.group.add(i),n={mesh:i,material:r,opacity:0,target:0},this.charges.set(e,n)}n.target=Pn(t.timeLeft,t.duration)}for(let e of this.charges.values()){let t=e.target>e.opacity?10:3;e.opacity+=(e.target-e.opacity)*(1-Math.exp(-r*t)),e.mesh.visible=e.opacity>.01;let i=e.material.uniforms;i.uOpacity.value=e.opacity*(.85+.15*Math.sin(n*9)),i.uTime.value=n}let i=+(this.aimRibbonRoute>=0);this.aimRibbonOpacity+=(i-this.aimRibbonOpacity)*(1-Math.exp(-r*14)),this.aimRibbon.uniforms.uOpacity.value=this.aimRibbonOpacity*.85,this.aimRibbon.uniforms.uTime.value=n,this.aimRibbonMeshes.forEach((e,t)=>{this.aimRibbonRoute>=0?e.visible=t===this.aimRibbonRoute:this.aimRibbonOpacity<.01&&(e.visible=!1)});for(let e of this.volleys.values())e.seen=!1;for(let e of t){let t=Vn(e),n=this.volleys.get(t);if(!n){n=this.volleyObject(t);let r=n.material.uniforms;r.uColor.value.setHex(R[e.side]),r.uLight.value.setHex(B[e.side]),rt(n.geometry,e.x,e.y,e.radius*Ln,this.heightAt,.18),n.wavesLeft=e.wavesLeft,this.volleys.set(t,n)}n.seen=!0,e.wavesLeft<n.wavesLeft&&(n.pulse=1),n.wavesLeft=e.wavesLeft;let r=Nn(e.nextIn,e.flight,e.wavesLeft,this.waves);n.material.uniforms.uAim.value=r.aim,n.material.uniforms.uLanding.value=+!!r.landing}for(let[e,t]of this.volleys){!t.seen&&t.wavesLeft>0&&(t.pulse=1),t.seen||(t.wavesLeft=0);let i=+!!t.seen;t.opacity+=(i-t.opacity)*(1-Math.exp(-r*(t.seen?12:2.5))),t.pulse=Math.max(0,t.pulse-r*3);let a=t.material.uniforms;a.uOpacity.value=t.opacity,a.uPulse.value=t.pulse,a.uTime.value=n,t.mesh.visible=t.opacity>.01,!t.seen&&t.opacity<.01&&(this.disposeVolley(t),this.volleys.delete(e))}let a=this.aimDisc;a.opacity+=(this.aimDiscTarget-a.opacity)*(1-Math.exp(-r*14)),a.mesh.visible=a.opacity>.01,a.material.uniforms.uOpacity.value=a.opacity,a.material.uniforms.uTime.value=n;for(let e=this.pennants.length-1;e>=0;e--){let t=this.pennants[e];t.t+=r;let i=Kt(Math.min(1,t.t/.6)),a=t.t>3.9?Math.max(0,(zn-t.t)/.6):1,o=Math.max(.01,i*a);for(let e of t.group.children){e.scale.set(1,o,1);let t=e.children[1];t.rotation.y=Math.sin(n*5+e.userData.phase)*.25}t.t>=zn&&(this.group.remove(t.group),this.pennants.splice(e,1))}}disposeVolley(e){this.group.remove(e.mesh),e.geometry.dispose(),e.material.dispose()}dispose(){for(let e of this.ribbons)e.dispose();for(let e of this.charges.values())e.material.dispose();this.aimRibbon.dispose();for(let e of this.volleys.values())this.disposeVolley(e);this.disposeVolley(this.aimDisc),this.pennantGeometry.dispose(),this.poleGeometry.dispose();for(let e of this.pennantMaterials)e.dispose();this.poleMaterial.dispose()}};function Vn(e){return(Math.round(e.x*8)*4096+Math.round(e.y*8))*2+e.side}var Hn=`
uniform vec3 uSunDir;
uniform vec3 uSunColor;
uniform vec3 uSkyColor;
uniform vec3 uGroundColor;
uniform vec3 uFogColor;
uniform float uFogDensity;

vec3 lambert(vec3 albedo, vec3 n) {
  float ndl = max(dot(n, uSunDir), 0.0);
  vec3 hemi = mix(uGroundColor, uSkyColor, n.y * 0.5 + 0.5);
  return albedo * (hemi + uSunColor * ndl) * 0.3183099;
}

vec3 applyFog(vec3 c, float dist) {
  float f = 1.0 - exp(-uFogDensity * uFogDensity * dist * dist);
  return mix(c, uFogColor, clamp(f, 0.0, 1.0));
}
`;function Un(e,t,n){let r=new x(16773596,2.9);r.position.copy(t).add(new A(-90,150,110)),r.target.position.copy(t),r.castShadow=!0,r.shadow.mapSize.set(4096,4096);let i=r.shadow.camera;i.left=-n,i.right=n,i.top=n,i.bottom=-n,i.near=10,i.far=500,r.shadow.bias=-4e-4,r.shadow.normalBias=.04,r.shadow.radius=6,r.shadow.blurSamples=12,e.add(r,r.target);let a=new h(14477813,9076578,1.55);e.add(a);let o=new le(H.fog,.0021);e.fog=o;let s={uSunDir:{value:new A},uSunColor:{value:new j},uSkyColor:{value:new j},uGroundColor:{value:new j},uFogColor:{value:new j},uFogDensity:{value:0},uTime:{value:0}},c=()=>{s.uSunDir.value.copy(r.position).sub(r.target.position).normalize(),s.uSunColor.value.copy(r.color).multiplyScalar(r.intensity),s.uSkyColor.value.copy(a.color).multiplyScalar(a.intensity),s.uGroundColor.value.copy(a.groundColor).multiplyScalar(a.intensity),s.uFogColor.value.copy(o.color),s.uFogDensity.value=o.density};return c(),{sun:r,hemi:a,fog:o,uniforms:s,sync:c}}var Wn=class{capacity;stride;data;next=0;firstDirty=0;written=0;emitted=0;constructor(e,t){this.capacity=e,this.stride=t,this.data=new Float32Array(e*t)}alloc(){let e=this.next;return this.written===0&&(this.firstDirty=e),this.written++,this.emitted++,this.next=(e+1)%this.capacity,e*this.stride}push(e){let t=this.alloc();for(let n=0;n<e.length;n++)this.data[t+n]=e[n];return t}takeRanges(){let e=this.written;if(this.written=0,e===0)return[];let t=this.stride;if(e>=this.capacity)return[[0,this.capacity*t]];let n=this.firstDirty+e;return n<=this.capacity?[[this.firstDirty*t,e*t]]:[[this.firstDirty*t,(this.capacity-this.firstDirty)*t],[0,(n-this.capacity)*t]]}},Gn=`
in vec4 a0;
in vec4 a1;
in vec4 a2;
in vec4 a3;
uniform float uTime;
const vec4 HIDDEN = vec4(2.0, 2.0, 2.0, 1.0);
mat3 axisAngle(vec3 axis, float a) {
  float c = cos(a), s = sin(a), t = 1.0 - c;
  vec3 n = normalize(axis);
  return mat3(
    t * n.x * n.x + c, t * n.x * n.y + s * n.z, t * n.x * n.z - s * n.y,
    t * n.x * n.y - s * n.z, t * n.y * n.y + c, t * n.y * n.z + s * n.x,
    t * n.x * n.z + s * n.y, t * n.y * n.z - s * n.x, t * n.z * n.z + c);
}
`,Kn=`
${Gn}
out vec3 vColor;
out vec3 vNormal;
out float vDist;
void main() {
  float t = uTime - a0.w;
  float life = a1.w;
  if (t < 0.0 || t > life) { gl_Position = HIDDEN; return; }
  float seed = a3.x;
  float k = a3.y;
  float g = a3.w;
  float e = (1.0 - exp(-k * t)) / k;
  vec3 p = a0.xyz + a1.xyz * e;
  p.y -= (g / k) * (t - e);
  bool landed = p.y <= a3.z + 0.02;
  vec3 axis = normalize(vec3(sin(seed * 12.9), cos(seed * 78.2), sin(seed * 37.7) + 0.2));
  mat3 r = landed ? axisAngle(vec3(1.0, 0.0, 0.0), -1.5708) * axisAngle(vec3(0.0, 0.0, 1.0), seed * 6.28)
                  : axisAngle(axis, t * (9.0 + seed * 8.0) + seed * 6.28);
  if (landed) p.y = a3.z + 0.02 + seed * 0.02;
  else p.x += sin(t * 7.0 + seed * 40.0) * 0.12;
  float s = a2.w * (1.0 - smoothstep(life - 0.6, life, t));
  vec3 local = r * vec3(position.xy * s, 0.0);
  vNormal = r * vec3(0.0, 0.0, 1.0);
  vColor = a2.rgb;
  vec4 mv = viewMatrix * vec4(p + local, 1.0);
  vDist = -mv.z;
  gl_Position = projectionMatrix * mv;
}
`,qn=`
${Hn}
in vec3 vColor;
in vec3 vNormal;
in float vDist;
void main() {
  vec3 n = normalize(vNormal);
  if (n.y < 0.0) n = -n;
  vec3 c = lambert(vColor, n) * 1.15;
  gl_FragColor = vec4(applyFog(c, vDist), 1.0);
  #include <colorspace_fragment>
}
`,Jn=`
${Gn}
out vec3 vColor;
out vec2 vUv;
out float vAlpha;
out float vDist;
void main() {
  float t = uTime - a0.w;
  float life = a1.w;
  if (t < 0.0 || t > life) { gl_Position = HIDDEN; return; }
  float k = max(a3.z, 0.01);
  float e = (1.0 - exp(-k * t)) / k;
  vec3 p = a0.xyz + a1.xyz * e;
  float u = t / life;
  float s = a2.w * (0.35 + a3.y * sqrt(u));
  vec4 mv = viewMatrix * vec4(p, 1.0);
  mv.xy += position.xy * s;
  vUv = position.xy;
  vColor = a2.rgb;
  vAlpha = a3.x * pow(1.0 - u, 1.4) * smoothstep(0.0, 0.08, u);
  vDist = -mv.z;
  gl_Position = projectionMatrix * mv;
}
`,Yn=`
${Hn}
in vec3 vColor;
in vec2 vUv;
in float vAlpha;
in float vDist;
void main() {
  float r = length(vUv);
  float a = vAlpha * (1.0 - smoothstep(0.55, 1.0, r));
  if (a < 0.004) discard;
  // Flat, slightly shaded disc: lighter on the upper-left, like a cel-shaded puff.
  float shade = 0.88 + 0.18 * smoothstep(0.2, -0.6, dot(vUv, vec2(0.6, -0.6)));
  vec3 c = lambert(vColor, vec3(0.0, 1.0, 0.0)) * shade;
  gl_FragColor = vec4(applyFog(c, vDist), a);
  #include <colorspace_fragment>
}
`,Xn=`
${Gn}
out vec3 vColor;
out vec3 vWorld;
out float vDist;
void main() {
  float t = uTime - a0.w;
  float flight = a1.w;
  float stick = a3.x;
  if (t < 0.0 || t > flight + stick) { gl_Position = HIDDEN; return; }
  float k = clamp(t / flight, 0.0, 1.0);
  vec3 from = a0.xyz;
  vec3 to = a1.xyz;
  float apex = a2.w;
  vec3 p = mix(from, to, k) + vec3(0.0, 4.0 * apex * k * (1.0 - k), 0.0);
  vec3 dir = (to - from) + vec3(0.0, 4.0 * apex * (1.0 - 2.0 * k), 0.0);
  dir = normalize(dir + vec3(0.0, 1e-4, 0.0));
  vec3 side = normalize(cross(vec3(0.0, 1.0, 0.0), dir) + vec3(1e-4, 0.0, 0.0));
  vec3 up = cross(dir, side);
  vec3 local = position * a3.y;
  vec3 world = p + side * local.x + up * local.y + dir * local.z;
  if (k >= 1.0) world -= dir * 0.25 * a3.y;
  vColor = a2.rgb;
  vWorld = world;
  vec4 mv = viewMatrix * vec4(world, 1.0);
  vDist = -mv.z;
  gl_Position = projectionMatrix * mv;
}
`,Zn=`
${Hn}
in vec3 vColor;
in vec3 vWorld;
in float vDist;
void main() {
  vec3 n = normalize(cross(dFdx(vWorld), dFdy(vWorld)));
  gl_FragColor = vec4(applyFog(lambert(vColor, n), vDist), 1.0);
  #include <colorspace_fragment>
}
`,Qn=`
${Gn}
out vec2 vUv;
out float vK;
out vec3 vColor;
out float vAlpha;
out float vEdge;
out float vDist;
void main() {
  float t = uTime - a0.w;
  float life = a1.z;
  if (t < 0.0 || t > life) { gl_Position = HIDDEN; return; }
  float u = t / life;
  float grow = 1.0 - pow(1.0 - min(1.0, u * a3.y), 3.0);
  float r = mix(a1.x, a1.y, grow);
  vec3 p = a0.xyz + vec3(position.x * r, 0.0, position.y * r);
  vUv = position.xy;
  vK = u;
  vColor = a2.rgb;
  vAlpha = a2.w * (1.0 - smoothstep(a3.z, 1.0, u));
  vEdge = a3.x;
  vec4 mv = viewMatrix * vec4(p, 1.0);
  vDist = -mv.z;
  gl_Position = projectionMatrix * mv;
}
`,$n=`
${Hn}
in vec2 vUv;
in float vK;
in vec3 vColor;
in float vAlpha;
in float vEdge;
in float vDist;
void main() {
  float r = length(vUv);
  if (r > 1.0) discard;
  // vEdge = 0: filled disc with soft rim (crater); > 0: ring of that relative width.
  float a = vEdge > 0.0
    ? smoothstep(1.0 - vEdge, 1.0 - vEdge * 0.5, r) * (1.0 - smoothstep(0.95, 1.0, r))
    : 1.0 - smoothstep(0.7, 1.0, r);
  a *= vAlpha;
  if (a < 0.004) discard;
  gl_FragColor = vec4(applyFog(vColor, vDist), a);
  #include <colorspace_fragment>
}
`,er=`
${Gn}
uniform float uFogDensity;
out vec3 vColor;
out vec2 vUv;
out float vAlpha;
void main() {
  float t = uTime - a0.w;
  float life = a1.w;
  if (t < 0.0 || t > life) { gl_Position = HIDDEN; return; }
  float k = max(a3.y, 0.01);
  float e = (1.0 - exp(-k * t)) / k;
  vec3 p = a0.xyz + a1.xyz * e;
  p.y -= 0.5 * a3.x * t * t;
  float seed = a3.z;
  p.x += sin(t * 3.1 + seed * 40.0) * a3.w;
  p.z += cos(t * 2.7 + seed * 23.0) * a3.w;
  float u = t / life;
  float s = a2.w * (1.0 - 0.55 * u);
  vec4 mv = viewMatrix * vec4(p, 1.0);
  mv.xy += position.xy * s;
  vUv = position.xy;
  vColor = a2.rgb;
  float flicker = 0.72 + 0.28 * sin(t * 17.0 + seed * 61.0);
  float d = -mv.z;
  float fog = exp(-uFogDensity * uFogDensity * d * d);
  vAlpha = smoothstep(0.0, 0.08, u) * (1.0 - smoothstep(0.55, 1.0, u)) * flicker * fog;
  gl_Position = projectionMatrix * mv;
}
`,tr=`
in vec3 vColor;
in vec2 vUv;
in float vAlpha;
void main() {
  float r = length(vUv);
  if (r > 1.0) discard;
  float core = 1.0 - smoothstep(0.0, 1.0, r);
  float glow = core * core + 0.6 * (1.0 - smoothstep(0.0, 0.32, r));
  gl_FragColor = vec4(vColor, min(1.0, glow * vAlpha));
  #include <colorspace_fragment>
}
`,nr=class{ring;mesh;buffer;geometry;material;constructor(e,t){this.ring=new Wn(e.capacity,16);for(let t=0;t<e.capacity;t++)this.ring.data[t*16+3]=-1e6;this.buffer=new l(this.ring.data,16,1),this.buffer.setUsage(ve);let n=new ye;n.index=e.geometry.index,n.setAttribute(`position`,e.geometry.attributes.position);for(let e=0;e<4;e++)n.setAttribute(`a${e}`,new w(this.buffer,4,e*4));n.instanceCount=e.capacity,this.geometry=n,this.material=new P({vertexShader:e.vertexShader,fragmentShader:e.fragmentShader,uniforms:{...t},transparent:e.transparent??!1,depthWrite:e.depthWrite??!0}),e.blending!==void 0&&(this.material.blending=e.blending),e.doubleSided&&(this.material.side=2),this.mesh=new v(n,this.material),this.mesh.frustumCulled=!1,this.mesh.name=e.name,this.mesh.renderOrder=e.renderOrder??0}flush(){let e=this.ring.takeRanges();if(e.length!==0){this.buffer.clearUpdateRanges();for(let[t,n]of e)this.buffer.addUpdateRange(t,n);this.buffer.needsUpdate=!0}}dispose(){this.geometry.dispose(),this.material.dispose()}},rr={confetti:24576,puffs:4096,arrows:3072,shells:256,shocks:512,decals:256,sparks:4096};function ir(){let e=new N(.035,.035,.85),n=new N(.16,.012,.16).translate(0,0,-.36),r=new N(.07,.07,.1).translate(0,0,.44);return t([e,n,r].map(e=>{let t=e.toNonIndexed();for(let e of Object.keys(t.attributes))e!==`position`&&t.deleteAttribute(e);return t}))}function ar(e,t=1){let n=Math.max(.05,Math.min(1,t)),r=Object.fromEntries(Object.entries(rr).map(([e,t])=>[e,Math.max(32,Math.round(t*n))])),i=new F(1,1),a=new F(2,2),o=new b(.22,0),s=o.index?o.toNonIndexed():o,c=new nr({name:`confetti`,capacity:r.confetti,geometry:i,vertexShader:Kn,fragmentShader:qn,doubleSided:!0},e),l=new nr({name:`puffs`,capacity:r.puffs,geometry:a,vertexShader:Jn,fragmentShader:Yn,transparent:!0,depthWrite:!1,renderOrder:4},e),u=new nr({name:`arrows`,capacity:r.arrows,geometry:ir(),vertexShader:Xn,fragmentShader:Zn},e),d=new nr({name:`shells`,capacity:r.shells,geometry:s,vertexShader:Xn,fragmentShader:Zn},e),f=new nr({name:`shocks`,capacity:r.shocks,geometry:a.clone(),vertexShader:Qn,fragmentShader:$n,transparent:!0,depthWrite:!1,doubleSided:!0,renderOrder:5},e),p=new nr({name:`decals`,capacity:r.decals,geometry:a.clone(),vertexShader:Qn,fragmentShader:$n,transparent:!0,depthWrite:!1,doubleSided:!0,renderOrder:1},e),m=new nr({name:`sparks`,capacity:r.sparks,geometry:a.clone(),vertexShader:er,fragmentShader:tr,transparent:!0,depthWrite:!1,blending:2,renderOrder:7},e);return s!==o&&o.dispose(),{confetti:c,puffs:l,arrows:u,shells:d,shocks:f,decals:p,sparks:m,all:[p,c,u,d,l,f,m]}}var or=5.2,sr=256,cr=128,lr=`
out vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`,ur=`
uniform vec3 uColor;
uniform float uFlash;
uniform float uTime;
uniform float uOpacity;
in vec2 vUv;
void main() {
  // Vertical bands drifting upwards, brighter at the bottom, plus the passage flash.
  float bands = 0.5 + 0.5 * sin(vUv.x * 30.0 + sin(vUv.y * 6.0 + uTime * 2.0) * 2.0);
  float rise = fract(vUv.y * 2.0 - uTime * 0.5);
  float fade = 1.0 - smoothstep(0.0, 1.0, vUv.y);
  float a = (0.16 + 0.14 * bands * rise) * fade;
  a += uFlash * 0.5 * (1.0 - vUv.y * 0.5);
  vec3 c = mix(uColor, vec3(1.0, 0.97, 0.85), uFlash * 0.7);
  gl_FragColor = vec4(c, min(0.85, a) * uOpacity);
  #include <colorspace_fragment>
}
`,dr=`
out vec2 vL;
void main() {
  vL = position.xz;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`,fr=`
uniform vec3 uColor;
uniform float uFlash;
uniform float uTime;
uniform float uOpacity;
uniform float uHalfWidth;
uniform float uHalfDepth;
uniform float uDir;
in vec2 vL;
void main() {
  vec2 q = abs(vL) / vec2(uHalfWidth, uHalfDepth);
  float border = smoothstep(0.86, 0.94, max(q.x, q.y));
  float s = vL.y * uDir - uTime * 2.2 - abs(vL.x) * 0.55;
  float chevron = fract(s / 1.8);
  float band = smoothstep(0.0, 0.08, chevron) * (1.0 - smoothstep(0.24, 0.32, chevron));
  float fade = 1.0 - smoothstep(0.92, 1.0, max(q.x, q.y));
  float a = (0.7 + band * 0.2 + border * 0.25) * fade;
  vec3 c = mix(uColor * 0.92, vec3(1.0, 0.98, 0.9), band * 0.45 + border * 0.8 + uFlash * 0.12);
  gl_FragColor = vec4(c, min(0.92, a) * uOpacity);
  #include <colorspace_fragment>
}
`;function pr(e){let t=new G;for(let n of[-e,e])t.add(new N(1.9,.5,1.9),H.stoneDark,K(n,.25,0)),t.add(new O(.55,.68,or,6),H.stone,K(n,2.9,0));return t.add(new N(2*e+1.2,.16,.95),V.gold,K(0,5.32,0)),t.build()}function mr(e){let t=new G().add(new N(2*e+1.6,.75,.8),16777215,K(0,5.78,0)).add(new N(2*e,.12,.9),16777215,K(0,.06,0));for(let n of[-e,e])t.add(new D(.85,1.5,4),16777215,K(n,6.9,0,Math.PI/4));let n=Math.max(2,Math.round(e/3));for(let r=0;r<n;r++){let i=-e+(r+.5)*2*e/n;t.add(new N(.9,1.5,.06),14211288,K(i,4.65,.45))}return t.build()}var hr=class{heightAt;group=new d;gates=new Map;stoneMaterial=new M({vertexColors:!0,flatShading:!0});time=0;ghost=null;ghostKey=``;onChange=null;constructor(e){this.heightAt=e,this.group.name=`gates`}build(e,t,n,r){let i=new d,a=pr(n),o=mr(n),s=new M({color:R[t],vertexColors:!0,flatShading:!0,emissive:new j(B[t]),emissiveIntensity:0,transparent:r,opacity:r?.55:1}),c=new v(a,r?s:this.stoneMaterial),l=new v(o,s);r||(c.castShadow=l.castShadow=!0,c.receiveShadow=l.receiveShadow=!0);let u=new F(2*n-1.2,4.9);u.translate(0,2.5,0);let f=new P({vertexShader:lr,fragmentShader:ur,transparent:!0,depthWrite:!1,side:2,uniforms:{uColor:{value:new j(B[t])},uFlash:{value:0},uTime:{value:0},uOpacity:{value:1}}}),p=new v(u,f);p.renderOrder=6;let m=2.2,h=new F(2*n,2*m).rotateX(-Math.PI/2);h.translate(0,.14,0);let g=new P({vertexShader:dr,fragmentShader:fr,transparent:!0,depthWrite:!1,uniforms:{uColor:{value:new j(R[t])},uFlash:{value:0},uTime:{value:0},uOpacity:{value:r?.7:1},uHalfWidth:{value:n},uHalfDepth:{value:m},uDir:{value:t===0?1:-1}}}),_=new v(h,g);_.renderOrder=2;let y=new dt(sr,cr),b=new de(new ce({map:y.texture,transparent:!0,depthWrite:!1,depthTest:!1}));return b.center.set(.5,0),b.renderOrder=10,b.position.set(0,6.5,0),i.add(c,l,p,_,b),{id:e,group:i,colourMaterial:s,veilMaterial:f,sillMaterial:g,plaque:b,label:y,flash:0,shakeUntil:0,seen:!0,leaving:0,appear:0,geometries:[a,o,u,h]}}place(e,t,n,r,i){let a=-i,o=r;e.group.position.set(t,this.heightAt(t,n),n),e.group.rotation.y=Math.atan2(-o,a)}flash(e){let t=this.gates.get(e);t&&(t.flash=1)}hit(e){let t=this.gates.get(e);t&&(t.shakeUntil=this.time+.12)}update(e,t,n,r){this.time=t;let i=!1;for(let e of this.gates.values())e.seen=!1;for(let t of e){let e=this.gates.get(t.id);e||(e=this.build(t.id,t.side,t.halfWidth,!1),this.place(e,t.x,t.y,t.tx,t.ty),this.gates.set(t.id,e),this.group.add(e.group),i=!0),e.seen=!0;let n=t.maxHp>0?t.hp/t.maxHp:null,r=`${Math.ceil(t.charges)}|${n===null?`-`:Math.round(n*40)}`;e.label.update(r,e=>pt(e,sr,cr,t.kind,t.charges,t.maxCharges,n,R[t.side],z[t.side]))}for(let[e,a]of this.gates){if(!a.seen&&(a.leaving+=n/.45,a.leaving>=1)){this.dispose1(a),this.gates.delete(e),i=!0;continue}a.appear=Math.min(1,a.appear+n/.35);let o=a.appear<1?1-(1-a.appear)**3:1,s=1-a.leaving*a.leaving;a.group.scale.set(1,Math.max(.01,o*s),1),this.animate(a,t,n,r)}this.ghost&&this.animate(this.ghost,t,n,r),i&&this.onChange?.()}animate(e,t,n,r){e.flash=Math.max(0,e.flash-n*6),e.veilMaterial.uniforms.uFlash.value=e.flash,e.veilMaterial.uniforms.uTime.value=t,e.sillMaterial.uniforms.uFlash.value=e.flash,e.sillMaterial.uniforms.uTime.value=t,e.colourMaterial.emissiveIntensity=e.flash*.3;let i=t<e.shakeUntil,a=e.group.children[0];a.position.x=i?(Math.random()-.5)*.3:0;let o=Math.max(1,r/70);e.plaque.scale.set(3.6*o,1.8*o,1)}setPreview(e,t){if(!e||!t){this.ghost&&(this.ghost.group.visible=!1);return}let n=`${e.kind}|${e.side}|${t.halfWidth.toFixed(1)}|${+!!e.valid}`;this.ghost&&n===this.ghostKey||(this.ghost&&this.dispose1(this.ghost),this.ghost=this.build(-1,e.side,t.halfWidth,!0),this.ghostKey=n,this.group.add(this.ghost.group),e.valid||(this.ghost.colourMaterial.color.setHex(9079434),this.ghost.colourMaterial.emissive.setHex(16726832),this.ghost.colourMaterial.emissiveIntensity=.35,this.ghost.veilMaterial.uniforms.uColor.value.setHex(16734799),this.ghost.sillMaterial.uniforms.uColor.value.setHex(9079434)),this.ghost.label.update(`ghost`,t=>pt(t,sr,cr,e.kind,0,0,null,R[e.side],z[e.side],!e.valid))),this.ghost.group.visible=!0,this.place(this.ghost,e.x,e.y,t.tx,t.ty),this.ghost.flash=e.valid?.25+.15*Math.sin(this.time*6):0}dispose1(e){this.group.remove(e.group);for(let t of e.geometries)t.dispose();e.colourMaterial.dispose(),e.veilMaterial.dispose(),e.sillMaterial.dispose(),e.plaque.material.dispose(),e.label.dispose()}dispose(){for(let e of this.gates.values())this.dispose1(e);this.gates.clear(),this.ghost&&this.dispose1(this.ghost),this.stoneMaterial.dispose()}},gr=`
in vec2 aRibbon; // across (-1..1), distance along the route (m)
out vec2 vR;
void main() {
  vR = aRibbon;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`,_r=`
uniform vec3 uColor;
uniform vec3 uEdge;
uniform float uTime;
uniform float uOpacity;
in vec2 vR;
void main() {
  float across = abs(vR.x);
  // Dashed edges along both borders.
  float dash = step(0.45, fract(vR.y / 2.2 - uTime * 0.6));
  float edge = smoothstep(0.8, 0.88, across) * (1.0 - smoothstep(0.97, 1.0, across));
  // Soft fill with a gentle shimmer travelling along the road.
  float shimmer = 0.5 + 0.5 * sin(vR.y * 0.35 - uTime * 2.4);
  float fill = (0.2 + 0.1 * shimmer) * (1.0 - smoothstep(0.75, 0.9, across));
  vec3 c = mix(uColor, uEdge, edge * dash);
  float a = (fill + edge * (0.35 + 0.6 * dash)) * uOpacity;
  gl_FragColor = vec4(c, a);
  #include <colorspace_fragment>
}
`,vr=class{heightAt;group=new d;routes;material;meshes=[];key=``;target=0;opacity=0;constructor(e,t){this.heightAt=t,this.group.name=`gate-zones`,this.routes=e.routes.map(e=>new o(e.key,e.name,e.points,e.widths)),this.material=new P({vertexShader:gr,fragmentShader:_r,transparent:!0,depthWrite:!1,uniforms:{uColor:{value:new j(16774872)},uEdge:{value:new j(16777215)},uTime:{value:0},uOpacity:{value:0}}})}set(e){let t=e??[];this.target=+(t.length>0);let n=t.map(e=>`${e.route}:${e.ranges.map(([e,t])=>`${e.toFixed(1)}-${t.toFixed(1)}`)}`).join(`|`);if(n!==this.key&&t.length!==0){this.key=n,this.clear();for(let e of t){let t=this.routes[e.route];if(t)for(let[n,r]of e.ranges){let e=this.ribbon(t,Math.max(0,n-1),Math.min(t.length,r+1));this.meshes.push(e),this.group.add(e)}}}}ribbon(e,t,n){let r=Math.max(2,Math.ceil((n-t)/1)+1),i=new Float32Array(r*2*3),a=new Float32Array(r*2*2),o=[];for(let s=0;s<r;s++){let c=t+(n-t)*s/(r-1),l=e.pointAt(c),u=e.tangentAt(c),d=e.widthAt(c)/2-.3;for(let e=0;e<2;e++){let t=e===0?-1:1,n=l.x-u.y*d*t,r=l.y+u.x*d*t,o=(s*2+e)*3;i[o]=n,i[o+1]=this.heightAt(n,r)+.16,i[o+2]=r,a[(s*2+e)*2]=t,a[(s*2+e)*2+1]=c}if(s>0){let e=(s-1)*2;o.push(e,e+1,e+2,e+1,e+3,e+2)}}let s=new re;s.setAttribute(`position`,new k(i,3)),s.setAttribute(`aRibbon`,new k(a,2)),s.setIndex(o);let c=new v(s,this.material);return c.renderOrder=2,c}clear(){for(let e of this.meshes)this.group.remove(e),e.geometry.dispose();this.meshes=[]}update(e,t){this.opacity+=(this.target-this.opacity)*(1-Math.exp(-t*14)),this.target===0&&this.opacity<.01&&this.meshes.length>0&&(this.clear(),this.key=``),this.material.uniforms.uOpacity.value=this.opacity,this.material.uniforms.uTime.value=e,this.group.visible=this.opacity>.01}dispose(){this.clear(),this.material.dispose()}},yr=`
in vec2 aRibbon; // across (-1..1), distance along the route (m)
out vec2 vR;
void main() {
  vR = aRibbon;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`,br=`
uniform vec3 uColor;
uniform float uTime;
uniform float uOpacity;
uniform float uDirection;
in vec2 vR;
void main() {
  float across = abs(vR.x);
  float edge = smoothstep(0.86, 0.95, across) * (1.0 - smoothstep(0.97, 1.0, across));
  float s = vR.y * uDirection - uTime * 6.0;
  float chevron = fract((s - across * 2.5) / 7.0);
  float band = smoothstep(0.0, 0.08, chevron) * (1.0 - smoothstep(0.22, 0.3, chevron));
  float a = (0.12 + edge * 0.75 + band * 0.4 * (1.0 - across)) * uOpacity;
  gl_FragColor = vec4(uColor, a);
  #include <colorspace_fragment>
}
`,xr=class{group=new d;meshes=[];material;active=null;opacity=0;constructor(e,t){this.group.name=`route-highlight`,this.material=new P({vertexShader:yr,fragmentShader:br,transparent:!0,depthWrite:!1,side:2,uniforms:{uColor:{value:new j(16774345)},uTime:{value:0},uOpacity:{value:0},uDirection:{value:1}}});for(let n of e.routes){let e=new o(n.key,n.name,n.points,n.widths),r=Math.ceil(e.length/1)+1,i=new Float32Array(r*2*3),a=new Float32Array(r*2*2),s=[];for(let n=0;n<r;n++){let r=Math.min(e.length,n*1),o=e.pointAt(r),c=e.tangentAt(r),l=e.widthAt(r)/2;for(let e=0;e<2;e++){let s=e===0?-1:1,u=o.x-c.y*l*s,d=o.y+c.x*l*s,f=(n*2+e)*3;i[f]=u,i[f+1]=t(u,d)+.12,i[f+2]=d,a[(n*2+e)*2]=s,a[(n*2+e)*2+1]=r}if(n>0){let e=(n-1)*2;s.push(e,e+1,e+2,e+1,e+3,e+2)}}let c=new re;c.setAttribute(`position`,new k(i,3)),c.setAttribute(`aRibbon`,new k(a,2)),c.setIndex(s);let l=new v(c,this.material);l.visible=!1,l.renderOrder=3,this.meshes.push(l),this.group.add(l)}}set(e,t=1){e===null?this.active=null:(this.active=e,this.material.uniforms.uDirection.value=t)}update(e,t){let n=this.active===null?0:1;this.opacity+=(n-this.opacity)*(1-Math.exp(-t*12)),this.material.uniforms.uOpacity.value=this.opacity,this.material.uniforms.uTime.value=e,this.meshes.forEach((e,t)=>{e.visible=this.opacity>.01&&(this.active===t||this.active===null&&e.visible)})}dispose(){for(let e of this.meshes)e.geometry.dispose();this.material.dispose()}};function Sr(){let e=new P({side:1,depthWrite:!1,fog:!1,uniforms:{uTop:{value:new j(H.skyTop)},uHorizon:{value:new j(H.skyHorizon)},uFog:{value:new j(H.fog)}},vertexShader:`
      out vec3 vDir;
      void main() {
        vDir = normalize(position);
        vec4 p = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        gl_Position = p.xyww;
      }`,fragmentShader:`
      uniform vec3 uTop;
      uniform vec3 uHorizon;
      uniform vec3 uFog;
      in vec3 vDir;
      void main() {
        float h = vDir.y;
        vec3 c = mix(uHorizon, uTop, smoothstep(0.0, 0.55, h));
        c = mix(uFog, c, smoothstep(-0.05, 0.08, h));
        gl_FragColor = vec4(c, 1.0);
        #include <colorspace_fragment>
      }`}),t=new v(new pe(800,24,12),e);return t.frustumCulled=!1,t.renderOrder=-10,t.name=`sky`,t}var Cr=class{positions=[];tags=[];indices=[];tmp=new A;get vertexCount(){return this.positions.length/3}get triangleCount(){return this.indices.length/3}vertex(e,t,n,r,i,a){let o=this.tmp.set(e,t,n).applyMatrix4(r);return this.positions.push(o.x,o.y,o.z),this.tags.push(i,a),this.vertexCount-1}quad(e,t,n,r){this.indices.push(e,t,n,e,n,r)}box(e,t,n,r,i,a,o={}){let s=t/2,c=n/2,l=r/2,u=o.topScale??1,d=(t,n,r)=>this.vertex(t,n,r,e,i,a),f=d(-s,-c,-l),p=d(s,-c,-l),m=d(s,-c,l),h=d(-s,-c,l),g=d(-s*u,c,-l*u),_=d(s*u,c,-l*u),v=d(s*u,c,l*u),y=d(-s*u,c,l*u);this.quad(h,m,v,y),this.quad(p,f,g,_),this.quad(m,p,_,v),this.quad(f,h,y,g),o.noTop||this.quad(y,v,_,g),o.noBottom||this.quad(f,p,m,h)}prism(e,t,n,r,i,a,o={top:!0,bottom:!0},s=1){let c=[],l=[];for(let o=0;o<t;o++){let u=o/t*Math.PI*2,d=Math.cos(u)*n,f=Math.sin(u)*n;c.push(this.vertex(d,-r/2,f,e,i,a)),l.push(this.vertex(d*s,r/2,f*s,e,i,a))}for(let e=0;e<t;e++){let n=(e+1)%t;this.quad(c[n],c[e],l[e],l[n])}if(o.top)for(let e=1;e<t-1;e++)this.indices.push(l[0],l[e+1],l[e]);if(o.bottom)for(let e=1;e<t-1;e++)this.indices.push(c[0],c[e],c[e+1])}pyramid(e,t,n,r,i,a){let o=this.vertex(0,r,0,e,i,a),s=[];for(let r=0;r<t;r++){let o=r/t*Math.PI*2+Math.PI/t;s.push(this.vertex(Math.cos(o)*n,0,Math.sin(o)*n,e,i,a))}for(let e=0;e<t;e++)this.indices.push(s[(e+1)%t],s[e],o)}build(){let e=new re;e.setAttribute(`position`,new k(new Float32Array(this.positions),3)),e.setAttribute(`aTag`,new k(new Float32Array(this.tags),2));let t=this.vertexCount>65535?new Uint32Array(this.indices):new Uint16Array(this.indices);return e.setIndex(new k(t,1)),e.computeBoundingSphere(),e}};function X(e,t,n,r=0,i=0,a=0,o){let s=new E().makeRotationX(r);return i&&s.premultiply(new E().makeRotationY(i)),a&&s.premultiply(new E().makeRotationZ(a)),s.premultiply(new E().makeTranslation(e,t,n)),o&&s.premultiply(o),s}var Z={Body:0,ArmRight:1,ArmLeft:2,Helmet:3,Cape:4,Barrel:5,Wheel:6,Base:7,LegFront:8,LegBack:9,Neck:10,Rider:11,Tail:12},Q={Side:0,SideDark:1,Skin:2,Trousers:3,Boots:4,Metal:5,Wood:6,Helmet:7,Iron:8,Belt:9,Straw:10,Canvas:11,Horse:12,Mane:13},wr=1.17,Tr=.27,Er=.36,Dr=-.05,Or=.68,kr=.33,Ar=.98,jr=.4,Mr=.22,Nr=1.42,Pr=-.06;[...xe];function Fr(e,t){t.base&&e.prism(X(0,.035,0,0,Math.PI/6),6,.42,.07,Z.Base,Q.SideDark,{top:!0}),t.legs&&(e.box(X(.1,.38,0),.15,.62,.18,Z.Body,Q.Trousers,{noTop:!0,noBottom:!0}),e.box(X(-.1,.38,0),.15,.62,.18,Z.Body,Q.Trousers,{noTop:!0,noBottom:!0})),e.box(X(0,.94,0),.46,.56,.28,Z.Body,Q.Side,{noBottom:!0,topScale:.9}),e.box(X(0,1.34,.01),.22,.24,.22,Z.Body,Q.Skin,{noBottom:!0})}function Ir(e,t){e.box(X(0,1.49,.01),.26,.14,.26,Z.Helmet,Q.Helmet,{noBottom:!0,topScale:.85}),t&&e.box(X(0,1.425,.05),.32,.03,.34,Z.Helmet,Q.Helmet,{noBottom:!0}),e.pyramid(X(0,1.56,.01),4,.06,.16,Z.Helmet,Q.Helmet)}function Lr(e){e.box(X(0,.8,-.2,-.1),.5,.78,.03,Z.Cape,Q.SideDark,{noTop:!0})}function Rr(e,t,n,r,i=0){let a=X(t*Tr,wr,0,r,0,i);e.box(X(0,-.22,0,0,0,0,a),.12,.5,.13,n,Q.Side,{noTop:!0})}function zr(){let e=new Cr;Fr(e,{base:!0,legs:!0}),Ir(e,!0),Lr(e),Rr(e,-1,Z.ArmRight,-.9),Rr(e,1,Z.ArmLeft,-.35);let t=X(0,-.46,0,.65,0,0,X(-.27,wr,0,-.9));return e.box(X(0,.04,0,0,0,0,t),.2,.04,.06,Z.ArmRight,Q.Iron,{noBottom:!0}),e.box(X(0,.34,0,0,0,0,t),.06,.56,.025,Z.ArmRight,Q.Metal,{noBottom:!0,topScale:.4}),e.prism(X(.36,.9,.16,0,0,Math.PI/2),6,.24,.04,Z.ArmLeft,Q.SideDark,{top:!0,bottom:!1}),e}function Br(){let e=new Cr;Fr(e,{base:!0,legs:!0}),Ir(e,!0),Lr(e),Rr(e,-1,Z.ArmRight,-.7),Rr(e,1,Z.ArmLeft,-.15);let t=X(-.27-.02,.86,.32,.32);return e.box(X(0,.45,0,0,0,0,t),.045,2.3,.045,Z.ArmRight,Q.Wood,{noTop:!0,noBottom:!0}),e.pyramid(X(0,1.6,0,0,Math.PI/4,0,t),4,.07,.34,Z.ArmRight,Q.Metal),e.box(X(0,1.35,-.12,0,0,0,t),.02,.16,.22,Z.ArmRight,Q.Side,{noTop:!0,noBottom:!0}),e}function Vr(){let e=new Cr;Fr(e,{base:!0,legs:!0}),Ir(e,!1),Lr(e),Rr(e,-1,Z.ArmRight,-1.2,.35),Rr(e,1,Z.ArmLeft,-1.35);let t=X(.25,1,.48);for(let[n,r,i]of[[0,0,.42],[.06,.38,.38],[.06,-.38,.38]]){let a=r>0?.35:r<0?-.35:0;e.box(X(0,r,n,a,0,0,t),.045,i,.045,Z.ArmLeft,Q.Wood,{noTop:!0,noBottom:!0})}return e.box(X(0,0,.13,0,0,0,t),.012,1.1,.012,Z.ArmLeft,Q.Trousers,{noTop:!0,noBottom:!0}),e.box(X(.1,1.02,-.2,.25,0,-.3),.12,.5,.12,Z.Body,Q.Belt,{noBottom:!0}),e}function Hr(){let e=new Cr,t=X(0,0,-.75);e.box(X(.1,.36,0,0,0,0,t),.15,.62,.18,Z.Body,Q.Trousers,{noTop:!0,noBottom:!0}),e.box(X(-.1,.36,0,0,0,0,t),.15,.62,.18,Z.Body,Q.Trousers,{noTop:!0,noBottom:!0}),e.box(X(0,.92,0,0,0,0,t),.46,.56,.28,Z.Body,Q.Side,{noBottom:!0,topScale:.9}),e.box(X(0,1.32,.01,0,0,0,t),.22,.24,.22,Z.Body,Q.Skin,{noBottom:!0}),e.box(X(0,1.47,.01,0,0,0,t),.26,.14,.26,Z.Helmet,Q.Helmet,{noBottom:!0,topScale:.85}),e.pyramid(X(0,1.54,.01,0,0,0,t),4,.06,.16,Z.Helmet,Q.Helmet),e.box(X(.26,.92,.2,-1.1,0,0,t),.12,.5,.13,Z.ArmLeft,Q.Side,{noTop:!0}),e.box(X(-.26,.92,.2,-1.1,0,0,t),.12,.5,.13,Z.ArmRight,Q.Side,{noTop:!0}),e.box(X(0,.78,-.95,-.1),.5,.74,.03,Z.Cape,Q.SideDark,{noTop:!0}),e.box(X(0,.42,-.1,.32),.34,.2,1.05,Z.Body,Q.Wood,{noBottom:!0}),e.prism(X(0,.66,.25,Math.PI/2-.12),6,.15,1.3,Z.Barrel,Q.Iron,{top:!1,bottom:!0},.8);for(let t of[-1,1])e.prism(X(t*.3,Er,Dr,0,0,Math.PI/2),6,.36,.08,Z.Wheel,Q.Wood,{top:!0,bottom:!0});return e}function Ur(){let e=new Cr;Fr(e,{base:!0,legs:!0}),e.box(X(0,.86,.15),.34,.5,.02,Z.Body,Q.Canvas,{noTop:!0,noBottom:!0}),e.box(X(0,1.475,.01),.5,.035,.5,Z.Body,Q.Straw),e.box(X(0,1.54,.01),.24,.12,.24,Z.Body,Q.Straw,{noBottom:!0,topScale:.8}),e.box(X(0,1,-.23),.32,.4,.16,Z.Body,Q.Canvas,{noBottom:!0}),Rr(e,-1,Z.ArmRight,-.7);let t=X(-.27-.03,1.24,.04,-.55);return e.box(X(0,0,0,0,0,0,t),.045,1,.045,Z.ArmRight,Q.Wood,{noTop:!0,noBottom:!0}),e.box(X(0,.5,0,0,0,0,t),.06,.07,.62,Z.ArmRight,Q.Iron,{topScale:.6}),Rr(e,1,Z.ArmLeft,-1.9),e.box(X(Tr,1.3,.05,-.1),.16,.05,1.8,Z.ArmLeft,Q.Wood),e}function Wr(){let e=new Cr;e.box(X(0,.035,0),.58,.07,1.36,Z.Base,Q.SideDark,{noBottom:!0,topScale:.95}),e.box(X(0,.84,0),.42,.42,1,Z.Body,Q.Horse,{topScale:.85});for(let[t,n,r]of[[kr,Z.LegFront,-.12],[-.33,Z.LegBack,.12]])for(let i of[-1,1]){let a=X(i*.12,Or,t,r);e.box(X(0,-.32,0,0,0,0,a),.11,.64,.13,n,Q.Horse,{noTop:!0,topScale:1.35})}let t=X(0,Ar,jr,.45);e.box(X(0,.27,0,0,0,0,t),.2,.6,.3,Z.Neck,Q.Horse,{noBottom:!0,topScale:.8}),e.box(X(0,.26,-.15,0,0,0,t),.05,.56,.07,Z.Neck,Q.Mane,{noTop:!0,noBottom:!0});let n=X(0,.54,.02,0,0,0,t);e.box(X(0,-.03,.17,.3,0,0,n),.19,.21,.48,Z.Neck,Q.Horse,{topScale:.9});for(let t of[-1,1])e.pyramid(X(t*.055,.06,-.03,-.45,0,0,n),3,.04,.13,Z.Neck,Q.Horse);let r=X(0,.98,-.49,.9);e.box(X(0,-.23,0,0,0,0,r),.15,.46,.07,Z.Tail,Q.Mane,{noTop:!0,topScale:.45}),e.box(X(0,.925,-.04),.48,.28,.5,Z.Body,Q.Side,{noBottom:!0});for(let t of[-1,1]){let n=X(t*.26,1.08,-.02,-.35);e.box(X(0,-.2,0,0,0,0,n),.12,.44,.14,Z.Rider,Q.Trousers,{noTop:!0})}e.box(X(0,1.27,-.06),.36,.42,.24,Z.Rider,Q.Side,{noBottom:!0,topScale:.9}),e.box(X(0,1.58,-.05),.19,.2,.19,Z.Rider,Q.Skin,{noBottom:!0}),e.box(X(0,1.73,-.05),.23,.12,.24,Z.Helmet,Q.Helmet,{noBottom:!0,topScale:.85}),e.box(X(0,1.82,-.08),.05,.1,.3,Z.Helmet,Q.Helmet,{noBottom:!0}),e.box(X(0,1.25,-.4,.93),.42,.56,.03,Z.Cape,Q.SideDark,{noTop:!0});let i=X(Mr,Nr,Pr,-.75);e.box(X(0,-.2,0,0,0,0,i),.1,.42,.11,Z.ArmLeft,Q.Side,{noTop:!0});let a=X(-.22,Nr,Pr,-1.25);e.box(X(0,-.2,0,0,0,0,a),.1,.42,.11,Z.ArmRight,Q.Side,{noTop:!0});let o=X(0,-.42,0,2.35,0,0,a);return e.box(X(0,.03,0,0,0,0,o),.18,.035,.05,Z.ArmRight,Q.Iron,{noBottom:!0}),e.box(X(0,.41,0,0,0,0,o),.05,.74,.02,Z.ArmRight,Q.Metal,{noBottom:!0,topScale:.35}),e}var Gr={sword:zr,spear:Br,bow:Vr,cannon:Hr,cavalry:Wr};function Kr(e){if(e===L)return Ur();let t=xe[e];if(!t)throw Error(`unknown unit kind ${e}`);return Gr[t]()}function qr(e){return Kr(e).build()}function Jr(e){let t=new Cr;return e===L?(t.box(X(0,.04,0),.8,.08,.8,Z.Base,Q.SideDark,{noBottom:!0}),t.box(X(0,.4,0),.36,.66,.2,Z.Body,Q.Trousers,{noTop:!0,noBottom:!0}),t.box(X(0,1.04,0),.5,.72,.3,Z.Body,Q.Side,{noBottom:!0}),t.box(X(0,1.48,0),.56,.12,.56,Z.Body,Q.Straw,{noBottom:!0}),t.box(X(Tr,1.32,.05,-.1),.18,.08,1.8,Z.ArmLeft,Q.Wood,{noBottom:!0}),t):e===I(`cavalry`)?(t.box(X(0,.04,0),.56,.08,1.3,Z.Base,Q.SideDark,{noBottom:!0}),t.box(X(0,.36,0),.26,.6,.8,Z.Body,Q.Horse,{noTop:!0,noBottom:!0}),t.box(X(0,.8,-.05),.38,.36,1,Z.Body,Q.Horse,{noBottom:!0}),t.box(X(0,1.1,.6,.9),.17,.62,.24,Z.Neck,Q.Horse,{noBottom:!0}),t.box(X(0,.86,-.05),.46,.3,.6,Z.Body,Q.Side,{noBottom:!0}),t.box(X(0,1.32,-.04),.44,.62,.28,Z.Rider,Q.Side,{noBottom:!0}),t.box(X(0,1.75,-.04),.26,.2,.28,Z.Helmet,Q.Helmet,{noBottom:!0}),t):e===I(`cannon`)?(t.box(X(0,.45,0,.1),.6,.5,1.5,Z.Body,Q.Iron,{noBottom:!0}),t.box(X(0,.95,-.75),.42,.9,.3,Z.Body,Q.Side,{noBottom:!0}),t.box(X(0,1.48,-.75),.26,.2,.26,Z.Helmet,Q.Helmet,{noBottom:!0}),t.box(X(0,.6,-.95),.5,.6,.04,Z.Cape,Q.SideDark,{noTop:!0,noBottom:!0}),t):(t.box(X(0,.04,0),.8,.08,.8,Z.Base,Q.SideDark,{noBottom:!0}),t.box(X(0,.4,0),.36,.66,.2,Z.Body,Q.Trousers,{noTop:!0,noBottom:!0}),t.box(X(0,1.04,0),.5,.72,.3,Z.Body,Q.Side,{noBottom:!0}),t.box(X(0,1.5,0),.28,.22,.28,Z.Helmet,Q.Helmet,{noBottom:!0}),t.box(X(0,.8,-.2),.5,.78,.04,Z.Cape,Q.SideDark,{noTop:!0,noBottom:!0}),e===I(`spear`)?t.box(X(-.3,1.3,.3,.32),.07,2.3,.07,Z.ArmRight,Q.Wood,{noTop:!0,noBottom:!0}):e===I(`bow`)?t.box(X(.25,1,.45),.06,1.1,.1,Z.ArmLeft,Q.Wood,{noTop:!0,noBottom:!0}):t.box(X(-.3,1.2,.3,-.4),.07,.6,.05,Z.ArmRight,Q.Metal,{noTop:!0,noBottom:!0}),t)}function Yr(e){return Jr(e).build()}var Xr=1.3,Zr=be,Qr=1024,$r=.55,ei=1<<17,ti=131071,ni=-1e6;function ri(e){let t=Math.PI*2,n=e%t;return n>Math.PI?n-=t:n<=-Math.PI&&(n+=t),n}function ii(e,t){return Math.atan2(e,t)}function ai(e,t,n,r,i){return e|t<<1|!!n<<3|!!r<<4|(i&1023)<<5}var oi=class{data=[];counts=new Int32Array(Zr);resized=Array(Zr).fill(!1);total=0;tblId=new Int32Array(ei).fill(-1);tblAngle=new Float32Array(ei);tblHeight=new Float32Array(ei);tblStrike=new Float32Array(ei);tblHit=new Float32Array(ei);tblHp=new Float32Array(ei);constructor(e=20480){for(let t=0;t<Zr;t++){let n=t===L?Math.min(e,Qr):e;this.data.push(new Float32Array(Math.max(1,n)*12))}}capacity(e){return this.data[e].length/12}grow(e){let t=this.data[e],n=new Float32Array(t.length*2);n.set(t),this.data[e]=n,this.resized[e]=!0}write(e,t,n){let r=this.counts;r.fill(0);let i=e.count,{tblId:a,tblAngle:o,tblHeight:s,tblStrike:c,tblHit:l,tblHp:u}=this;for(let d=0;d<i;d++){let i=e.weapon[d];if(i>=Zr)continue;r[i]>=this.capacity(i)&&this.grow(i);let f=this.data[i],p=r[i]*12;r[i]++;let m=e.id[d],h=e.x[d],g=e.y[d],_=e.prevX[d],v=e.prevY[d],y=e.flags[d],b=e.hp[d],x=m&ti,S=ii(e.fx[d],e.fy[d]),C=n(h,g),w,T;a[x]===(m|0)?(w=o[x],T=s[x]):(a[x]=m|0,w=S,T=_===h&&v===g?C:n(_,v),c[x]=ni,l[x]=ni,u[x]=b);let E=w+ri(S-w)*$r;o[x]=E,s[x]=C;let ee=!!(y&1);ee&&(c[x]=t),b<u[x]-1e-4&&(l[x]=t),u[x]=b,f[p]=_,f[p+1]=v,f[p+2]=h,f[p+3]=g,f[p+4]=T,f[p+5]=C,f[p+6]=w,f[p+7]=E,f[p+8]=e.mult[d]>1?Xr:1,f[p+9]=ai(e.side[d],e.level[d],ee,!!(y&2),m),f[p+10]=c[x],f[p+11]=l[x]}this.total=i}},si=I(`sword`),ci=I(`spear`),li=I(`bow`),ui=I(`cannon`),di=I(`cavalry`),$=e=>Number.isInteger(e)?`${e}.0`:`${e}`,fi=`
in vec4 iA; // start x, start y, end x, end y
in vec4 iB; // start h, end h, start angle, end angle
in vec4 iC; // scale, packed bits, strike time, hit time

uniform float uAlpha;
uniform float uStepDt;
uniform float uTime;

const float PI = 3.14159265;

vec2 instancePos() { return mix(iA.xy, iA.zw, uAlpha); }
float instanceHeight() { return mix(iB.x, iB.y, uAlpha); }
float instanceAngle() {
  float d = iB.w - iB.z;
  d = mod(d + PI, 2.0 * PI) - PI;
  return iB.z + d * uAlpha;
}
float instanceSpeed() { return length(iA.zw - iA.xy) / max(uStepDt, 1e-3); }
`,pi=`
${fi}
in vec2 aTag;

uniform vec3 uSide[2];
uniform vec3 uSideDark[2];
uniform vec3 uPalette[14];
uniform vec3 uGold;
uniform vec3 uIron;

out vec3 vColor;
out vec3 vWorld;
out float vDist;

mat3 rotX(float a) { float c = cos(a), s = sin(a); return mat3(1.0, 0.0, 0.0, 0.0, c, s, 0.0, -s, c); }
mat3 rotY(float a) { float c = cos(a), s = sin(a); return mat3(c, 0.0, -s, 0.0, 1.0, 0.0, s, 0.0, c); }
mat3 rotZ(float a) { float c = cos(a), s = sin(a); return mat3(c, s, 0.0, -s, c, 0.0, 0.0, 0.0, 1.0); }

// Bell curve on [0, 1] (0 at both ends, 1 in the middle).
float bell(float k) { return k <= 0.0 || k >= 1.0 ? 0.0 : sin(k * PI); }

void main() {
  int part = int(aTag.x + 0.5);
  int slot = int(aTag.y + 0.5);
  int packed = int(iC.y + 0.5);
  int side = packed & 1;
  int level = (packed >> 1) & 3;
  bool engaged = ((packed >> 4) & 1) == 1;
  float phase = float(packed >> 5) / 1024.0;

  vec3 pos = position;
  if (part == ${Z.Cape} && level < 2) pos = vec3(0.0);

  float speed = instanceSpeed();
  float moving = clamp(speed / 2.5, 0.0, 1.0);
  float cyc = uTime * 9.0 + phase * 6.2831;
  float strikeK = (uTime - iC.z) / 0.42;
  float hitK = (uTime - iC.w) / 0.3;
  float strike = bell(strikeK);

  vec3 shoulderR = vec3(${$(-Tr)}, ${$(wr)}, 0.0);
  vec3 shoulderL = vec3(${$(Tr)}, ${$(wr)}, 0.0);

#if WEAPON == ${si}
  // Sword: raise, then chop down and forward.
  if (part == ${Z.ArmRight}) {
    float a = 0.0;
    if (strikeK > 0.0 && strikeK < 1.0) {
      a = strikeK < 0.35
        ? -1.3 * (strikeK / 0.35)
        : mix(-1.3, 0.8, smoothstep(0.35, 0.6, strikeK)) * (1.0 - smoothstep(0.6, 1.0, strikeK));
    } else if (engaged) {
      a = -0.15 + 0.15 * sin(cyc * 0.7);
    }
    pos = rotX(a) * (pos - shoulderR) + shoulderR;
  }
  if (part == ${Z.ArmLeft} && engaged) pos = rotX(-0.35) * (pos - shoulderL) + shoulderL;
#elif WEAPON == ${ci}
  // Spear: lower to the front and thrust.
  if (part == ${Z.ArmRight}) {
    float lower = engaged || strikeK < 1.0 ? 0.55 : 0.0;
    pos = rotX(lower) * (pos - vec3(-0.27, 0.86, 0.32)) + vec3(-0.27, 0.86, 0.32);
    pos.z += 0.55 * strike;
  }
#elif WEAPON == ${li}
  // Bow: the drawing arm snaps back at release, the bow kicks up.
  if (part == ${Z.ArmRight}) pos = rotX(0.6 * strike) * (pos - shoulderR) + shoulderR;
  if (part == ${Z.ArmLeft}) pos = rotX(-0.25 * strike) * (pos - shoulderL) + shoulderL;
#elif WEAPON == ${ui}
  // Cannon: the barrel recoils, wheels roll with the march.
  if (part == ${Z.Barrel}) pos.z -= 0.35 * bell(clamp(strikeK * 1.6, 0.0, 1.0));
  if (part == ${Z.Wheel}) {
    vec3 axle = vec3(0.0, ${$(Er)}, ${$(Dr)});
    pos = rotX(uTime * speed / 0.36) * (pos - axle) + axle;
  }
#elif WEAPON == ${di}
  // Cavalry: gallop (leg pairs swung against each other, neck nodding, rider posting),
  // a sabre cut from the saddle at the strike, a restless stamp when held in a melee.
  float gal = uTime * ${$(14)} + phase * 6.2831;
  float stride = sin(gal);
  float stamp = engaged ? (1.0 - moving) * 0.3 * sin(uTime * 7.0 + phase * 6.2831) : 0.0;
  if (part == ${Z.LegFront} || part == ${Z.LegBack}) {
    bool front = part == ${Z.LegFront};
    // Front pair reaches forward while the hind pair kicks back, then both gather.
    float a = (front ? -0.62 : 0.55) * moving * stride + (front ? stamp * sign(pos.x) : 0.0);
    // A leg off the ground folds a little (lifted hoof).
    vec3 pivot = vec3(0.0, ${$(Or)}, front ? ${$(kr)} : ${$(-kr)});
    pos = rotX(a) * (pos - pivot) + pivot;
    pos.y += 0.05 * moving * max(0.0, front ? -stride : stride) * (1.0 - clamp(pos.y / 0.66, 0.0, 1.0));
  }
  if (part == ${Z.Neck}) {
    vec3 withers = vec3(0.0, ${$(Ar)}, ${$(jr)});
    float nod = 0.16 * moving * sin(gal - 1.2) + 0.06 * strike + 0.5 * stamp * stamp;
    pos = rotX(nod) * (pos - withers) + withers;
  }
  if (part == ${Z.Tail}) pos.x += 0.06 * sin(uTime * 3.0 + phase * 9.0) * (0.92 - pos.y);
  bool rider = part == ${Z.Rider} || part == ${Z.Helmet} || part == ${Z.Cape}
    || part == ${Z.ArmRight} || part == ${Z.ArmLeft};
  vec3 shoulderRR = vec3(${$(-Mr)}, ${$(Nr)}, ${$(Pr)});
  vec3 shoulderRL = vec3(${$(Mr)}, ${$(Nr)}, ${$(Pr)});
  if (part == ${Z.ArmRight}) {
    // Sabre: raised high over the head, then cut down past the horse's neck.
    float a = 0.0;
    if (strikeK > 0.0 && strikeK < 1.0) {
      a = strikeK < 0.3
        ? -1.5 * smoothstep(0.0, 0.3, strikeK)
        : mix(-1.5, 1.0, smoothstep(0.3, 0.5, strikeK)) * (1.0 - smoothstep(0.55, 1.0, strikeK));
    } else if (engaged) {
      a = -0.3 + 0.25 * sin(uTime * 3.1 + phase * 6.2831);
    }
    pos = rotX(a) * (pos - shoulderRR) + shoulderRR;
  }
  if (part == ${Z.ArmLeft}) pos = rotX(0.12 * moving * stride) * (pos - shoulderRL) + shoulderRL;
  if (rider) {
    // The rider rises and sits back once per stride, leaning into the cut.
    pos.y += 0.035 * moving * sin(gal * 2.0 - 0.8);
    vec3 seat = vec3(0.0, 1.065, -0.06);
    pos = rotX(0.12 * strike - 0.05 * moving * stride) * (pos - seat) + seat;
  }
#elif WEAPON == ${L}
  // Engineer: caught by an enemy, it cowers with its tools raised over its head and
  // trembles; standing still (waiting at a stalled site), it swings its pickaxe.
  if (engaged) {
    float tremble = 0.07 * sin(uTime * 41.0 + phase * 30.0);
    if (part == ${Z.ArmRight}) pos = rotX(-0.75 + tremble) * (pos - shoulderR) + shoulderR;
    if (part == ${Z.ArmLeft}) pos = rotX(0.5 - tremble) * (pos - shoulderL) + shoulderL;
    pos.y *= 0.9;
  } else if (part == ${Z.ArmRight}) {
    float work = (1.0 - moving) * pow(max(0.0, sin(uTime * 4.2 + phase * 6.2831)), 3.0);
    float sway = 0.12 * moving * sin(cyc);
    pos = rotX(-0.9 * work + 1.1 * work * work + sway) * (pos - shoulderR) + shoulderR;
  }
#endif

#if WEAPON == ${di}
  if (part == ${Z.Cape}) {
    // Flying back over the rump, flapping at the gallop.
    float k = clamp((${$(Nr)} - pos.y) / 0.4, 0.0, 1.0);
    pos.z -= k * 0.14 * moving;
    pos.y += k * (0.1 * moving * sin(gal * 2.0 + pos.x * 4.0) + 0.04 * sin(cyc * 0.5));
  }
#else
  if (part == ${Z.Cape}) {
    float k = clamp((1.2 - pos.y) / 0.8, 0.0, 1.0);
    pos.z -= k * (0.28 * moving + 0.05 * sin(cyc * 0.5 + pos.x * 3.0));
  }
#endif

#if WEAPON == ${ui}
  float hop = 0.02 * moving * abs(sin(cyc));
  float roll = 0.03 * moving * sin(cyc);
  float lean = 0.0;
#elif WEAPON == ${di}
  // Rocking-horse gallop: the whole figure bounds once per stride and pitches nose up
  // as the front legs reach out, nose down as they land; a heavy horse barely rolls.
  float hop = 0.09 * moving * max(0.0, sin(gal + 0.6));
  float roll = 0.025 * moving * sin(gal * 0.5);
  float lean = -0.09 * moving * stride + 0.05 * strike;
#elif WEAPON == ${L}
  // Loaded with tools: a heavier, shorter hop, leaning back when caught.
  float hop = 0.08 * moving * abs(sin(cyc));
  float roll = 0.09 * moving * sin(cyc);
  float lean = 0.13 * moving - (engaged ? 0.22 : 0.0);
#else
  float hop = 0.11 * moving * abs(sin(cyc));
  float roll = 0.12 * moving * sin(cyc);
  float lean = 0.1 * moving + (engaged ? 0.06 * strike : 0.0);
#endif
  float recoil = bell(hitK);
#if WEAPON == ${di}
  // A horse is heavy: it shies rather than topples.
  lean -= 0.16 * recoil;
  pos.z -= 0.1 * recoil;
#else
  lean -= 0.32 * recoil;
  pos.z -= 0.14 * recoil;
#endif

  pos = rotZ(roll) * rotX(lean) * pos;
  pos.y += hop;

  pos *= iC.x;
  pos = rotY(instanceAngle()) * pos;
  vec2 p = instancePos();
  vec3 world = vec3(p.x, instanceHeight(), p.y) + pos;

  vec3 c;
  if (slot == ${Q.Side}) c = uSide[side];
  else if (slot == ${Q.SideDark}) c = uSideDark[side];
  else if (slot == ${Q.Helmet}) c = level >= 1 ? uGold : uIron;
  else c = uPalette[slot];
  vColor = c;
  vWorld = world;
  vec4 mv = viewMatrix * vec4(world, 1.0);
  vDist = -mv.z;
  gl_Position = projectionMatrix * mv;
}
`,mi=`
${Hn}
in vec3 vColor;
in vec3 vWorld;
in float vDist;

void main() {
  vec3 n = normalize(cross(dFdx(vWorld), dFdy(vWorld)));
  vec3 c = lambert(vColor, n);
  // Rim of light from the sky on top surfaces keeps figures crisp at a distance.
  c += vColor * 0.08 * smoothstep(0.6, 1.0, n.y);
  gl_FragColor = vec4(applyFog(c, vDist), 1.0);
  #include <colorspace_fragment>
}
`,hi=`
${fi}
uniform vec2 uShadowSize; // lateral, along the facing direction
uniform vec2 uShadowOffset;
out vec2 vUv;
void main() {
  vUv = position.xz;
  vec2 p = instancePos();
  float a = instanceAngle();
  float c = cos(a), sn = sin(a);
  // Elongated along the facing direction (cannons), offset away from the sun.
  vec2 local = vec2(position.x * uShadowSize.x, position.z * uShadowSize.y) * iC.x;
  vec2 rotated = vec2(local.x * c + local.y * sn, -local.x * sn + local.y * c);
  vec3 world = vec3(p.x + rotated.x + uShadowOffset.x * iC.x, instanceHeight() + 0.035, p.y + rotated.y + uShadowOffset.y * iC.x);
  gl_Position = projectionMatrix * viewMatrix * vec4(world, 1.0);
}
`,gi=`
uniform float uShadowOpacity;
in vec2 vUv;
void main() {
  float r = length(vUv);
  float a = (1.0 - smoothstep(0.35, 1.0, r)) * uShadowOpacity;
  if (a < 0.003) discard;
  gl_FragColor = vec4(0.12, 0.14, 0.2, a);
}
`;function _i(e){if(e===L)return[.55,.75];switch(xe[e]){case`sword`:return[.55,.55];case`cannon`:return[.75,1.25];case`cavalry`:return[.5,1.05];default:return[.55,.6]}}function vi(e){let t=new ye;t.index=e.index;for(let[n,r]of Object.entries(e.attributes))t.setAttribute(n,r);return t.instanceCount=0,t}var yi=class{group=new d;writer=new oi;draws=[];frameUniforms={uAlpha:{value:1},uStepDt:{value:1/30}};lodActive=!1;materials=[];constructor(e){let t=e=>new j(e),n=[],r={[Q.Side]:R[0],[Q.SideDark]:z[0],[Q.Skin]:V.skin,[Q.Trousers]:V.trousers,[Q.Boots]:V.boots,[Q.Metal]:V.metal,[Q.Wood]:V.wood,[Q.Helmet]:V.iron,[Q.Iron]:V.iron,[Q.Belt]:V.belt,[Q.Straw]:V.straw,[Q.Canvas]:V.canvas,[Q.Horse]:V.horse,[Q.Mane]:V.mane};for(let e=0;e<14;e++)n.push(t(r[e]??16711935));let i={uSide:{value:[t(R[0]),t(R[1])]},uSideDark:{value:[t(z[0]),t(z[1])]},uPalette:{value:n},uGold:{value:t(V.gold)},uIron:{value:t(V.iron)}},a=new F(2,2).rotateX(-Math.PI/2);for(let t=0;t<Zr;t++){let n=new P({vertexShader:pi,fragmentShader:mi,defines:{WEAPON:t},uniforms:{...e,...this.frameUniforms,...i}});this.materials.push(n);let r=new P({vertexShader:hi,fragmentShader:gi,transparent:!0,depthWrite:!1,uniforms:{...e,...this.frameUniforms,uShadowSize:{value:_i(t)},uShadowOffset:{value:[.18,.12]},uShadowOpacity:{value:.36}}});this.materials.push(r);let o=new l(this.writer.data[t],12,1);o.setUsage(ve);let s=vi(qr(t)),c=vi(Yr(t)),u=vi(a);for(let e of[s,c,u])this.bindInstances(e,o);let d=new v(s,n);d.frustumCulled=!1,d.name=`soldiers-${t}`;let f=new v(u,r);f.frustumCulled=!1,f.renderOrder=1,f.name=`soldier-shadows-${t}`,this.group.add(d,f),this.draws.push({buffer:o,full:s,lod:c,shadowGeo:u,mesh:d,shadow:f})}}bindInstances(e,t){e.setAttribute(`iA`,new w(t,4,0)),e.setAttribute(`iB`,new w(t,4,4)),e.setAttribute(`iC`,new w(t,4,8))}setSoldiers(e,t,n){this.writer.write(e,t,n);for(let e=0;e<Zr;e++){let t=this.draws[e],n=this.writer.counts[e];if(this.writer.resized[e]){this.writer.resized[e]=!1,t.buffer=new l(this.writer.data[e],12,1),t.buffer.setUsage(ve);for(let e of[t.full,t.lod,t.shadowGeo])this.bindInstances(e,t.buffer)}t.buffer.clearUpdateRanges(),n>0&&(t.buffer.addUpdateRange(0,n*12),t.buffer.needsUpdate=!0),t.full.instanceCount=n,t.lod.instanceCount=n,t.shadowGeo.instanceCount=n,t.mesh.visible=n>0,t.shadow.visible=n>0}}update(e,t,n){this.frameUniforms.uAlpha.value=e,this.frameUniforms.uStepDt.value=t;let r=n>95;if(r!==this.lodActive){this.lodActive=r;for(let e of this.draws)e.mesh.geometry=r?e.lod:e.full}}get count(){return this.writer.total}get usingLod(){return this.lodActive}dispose(){for(let e of this.draws)e.full.dispose(),e.lod.dispose(),e.shadowGeo.dispose();for(let e of this.materials)e.dispose()}},bi=.82,xi=2,Si=16;function Ci(e,t,n){let r=1/0;for(let i of e){let e=i.project(t,n),a=Math.abs(e.offset)-i.widthAt(e.distance)/2*bi;a<r&&(r=a)}return r}function wi(e){let t=e.map,n=t.width*xi,r=t.height*xi,i=t.routes.map(e=>new o(e.key,e.name,e.points,e.widths)),a=new Uint8Array(n*r);for(let e=0;e<r;e++)for(let t=0;t<n;t++){let r=Ci(i,(t+.5)/xi,(e+.5)/xi);a[e*n+t]=Math.max(0,Math.min(255,Math.round(128+r/Si*127)))}let s=new ge(a,n,r,y,S);return s.magFilter=f,s.minFilter=f,s.wrapS=ie,s.wrapT=ie,s.flipY=!1,s.needsUpdate=!0,s}var Ti=class{info;texture;display;target;data;initialised=!1;constructor(e){this.info=e;let t=e.cols*e.rows;this.display=new Float32Array(t),this.target=new Float32Array(t),this.data=new Uint8Array(t*4);for(let n=0;n<t;n++)this.data[n*4+1]=e.mask[n]?255:0,this.data[n*4+3]=255;this.texture=new ge(this.data,e.cols,e.rows,ne,S),this.texture.magFilter=f,this.texture.minFilter=f,this.texture.flipY=!1,this.texture.needsUpdate=!0}setTarget(e){this.target.set(e.subarray(0,this.target.length)),this.initialised||=(this.display.set(this.target),!0)}update(e){let t=1-Math.exp(-e*3.5),{display:n,target:r,data:i}=this;for(let e=0;e<n.length;e++){let a=n[e]+(r[e]-n[e])*t;n[e]=a,i[e*4]=Math.round((a+1)*127.5)}this.texture.needsUpdate=!0}dispose(){this.texture.dispose()}},Ei=3,Di=(e,t)=>ct(e,t)*.55+ct(e*2.13+7.1,t*2.13+7.1)*.3+ct(e*4.7-3.3,t*4.7-3.3)*.15,Oi=(e,t,n)=>{let r=Math.min(1,Math.max(0,(n-e)/(t-e)));return r*r*(3-2*r)};function ki(e){let t=e.bounds,n=Math.round((t.x1-t.x0)*Ei),r=Math.round((t.y1-t.y0)*Ei),i=new Uint8Array(n*r*4),a=new Uint8Array(n*r*2),o=e=>[e>>16&255,e>>8&255,e&255],s=o(H.grass),c=o(H.meadow).map(e=>e*.94),l=o(H.sand),u=o(H.earth),d=o(H.rock),h=o(H.rockDark),g=e.map.width,_=e.map.height,v=[0,0,0],y=1,b=(e,t)=>{y*=1-t,v[0]+=(e[0]-v[0])*t,v[1]+=(e[1]-v[1])*t,v[2]+=(e[2]-v[2])*t};for(let o=0;o<r;o++){let r=t.y0+(o+.5)/Ei;for(let f=0;f<n;f++){let p=t.x0+(f+.5)/Ei,m=e.groundHeight(p,r);v[0]=s[0],v[1]=s[1],v[2]=s[2],y=1,b(c,.65*(1-(1-Oi(0,10,Math.max(-p,p-g,-r,r-_)+(ct(p*.3,r*.3)-.5)*6)))),b(l,1-Oi(-.32,-.12,m)),b(u,1-Oi(-.9,-.45,m));let x=Oi(4.6,5,m+(ct(p*.5,r*.5)-.5)*1.4);b(d,x),b(h,x*Oi(5.6,6.4,m)*.6);let S=(o*n+f)*4;i[S]=v[0],i[S+1]=v[1],i[S+2]=v[2],i[S+3]=Math.round(y*255);let C=(o*n+f)*2;a[C]=Math.round(Di(p*.045,r*.045)*255),a[C+1]=Math.round(Di(p*.16,r*.16)*255)}}let x=e=>(e.magFilter=f,e.minFilter=m,e.generateMipmaps=!0,e.anisotropy=8,e.flipY=!1,e.needsUpdate=!0,e),C=x(new ge(i,n,r,ne,S));C.colorSpace=he;let w=x(new ge(a,n,r,p,S));return w.unpackAlignment=2,{base:C,noise:w}}function Ai(e,t,n){let{bounds:r}=e,i=r.x1-r.x0,a=r.y1-r.y0,o=new F(i,a,Math.round(i),Math.round(a));o.rotateX(-Math.PI/2),o.translate(r.x0+i/2,0,r.y0+a/2);let s=o.attributes.position;for(let t=0;t<s.count;t++)s.setY(t,e.groundHeight(s.getX(t),s.getZ(t)));o.computeVertexNormals(),o.computeBoundingSphere();let l=wi(t),{base:u,noise:d}=ki(e),f=e=>{let t=new j(e),n=new j(H.grass);return new A(t.r/n.r,t.g/n.g,t.b/n.b)},p=t.map,m=t.territory,h={uBase:{value:u},uNoise:{value:d},uDarkK:{value:f(H.grassDark)},uLightK:{value:f(H.grassLight)},uBounds:{value:new g(r.x0,r.y0,i,a)},uRoad:{value:l},uMapSize:{value:new c(p.width,p.height)},uTerritory:{value:n.texture},uTerritorySize:{value:new c(m.cols*m.cellSize,m.rows*m.cellSize)},uInkBlue:{value:new j(He[0])},uInkRed:{value:new j(He[1])},uRoadColor:{value:new j(H.road)},uRoadEdge:{value:new j(H.roadEdge)}},_=new M({color:16777215,flatShading:!0});_.onBeforeCompile=e=>{Object.assign(e.uniforms,h),e.vertexShader=e.vertexShader.replace(`#include <common>`,`#include <common>
varying vec3 vGround;`).replace(`#include <project_vertex>`,`#include <project_vertex>
vGround = (modelMatrix * vec4(transformed, 1.0)).xyz;`),e.fragmentShader=e.fragmentShader.replace(`#include <common>`,`#include <common>
varying vec3 vGround;
uniform sampler2D uBase;
uniform sampler2D uNoise;
uniform vec3 uDarkK, uLightK;
uniform vec4 uBounds;
uniform sampler2D uRoad;
uniform vec2 uMapSize;
uniform sampler2D uTerritory;
uniform vec2 uTerritorySize;
uniform vec3 uInkBlue, uInkRed, uRoadColor, uRoadEdge;
vec3 groundColor(vec3 p) {
  vec2 xz = p.xz;
  vec2 buv = (xz - uBounds.xy) / uBounds.zw;
  vec4 baked = texture2D(uBase, buv);
  vec2 nz = texture2D(uNoise, buv).rg;
  float grain = nz.g;
  // Flat patches of grass: three tones, antialiased thresholds.
  float pa = fwidth(nz.r) + 1e-4;
  float tDark = 1.0 - smoothstep(0.42 - pa, 0.42 + pa, nz.r);
  float tLight = smoothstep(0.6 - pa, 0.6 + pa, nz.r);
  vec3 tone = mix(mix(vec3(1.0), uDarkK, tDark), uLightK, tLight);
  vec3 c = baked.rgb * mix(vec3(1.0), tone, baked.a);
  vec2 outside = max(-xz, xz - uMapSize);
  float inMap = 1.0 - smoothstep(-2.0, 6.0, max(outside.x, outside.y));
  float wet = 1.0 - smoothstep(-0.32, -0.12, p.y);
  float rock = smoothstep(4.6, 5.2, p.y);
  // Roads from the signed-distance texture (crisp at every zoom).
  float rd = (texture2D(uRoad, xz / uMapSize).r * 255.0 - 128.0) / 127.0 * ${Si.toFixed(1)};
  rd += (grain - 0.5) * 0.7;
  float aa = max(fwidth(rd), 0.02);
  float road = (1.0 - smoothstep(-aa, aa, rd)) * inMap * (1.0 - rock) * (1.0 - wet * 0.6);
  float edge = (1.0 - smoothstep(-aa, aa, abs(rd + 0.35) - 0.35)) * inMap * (1.0 - rock);
  c = mix(c, uRoadColor, road);
  c = mix(c, uRoadEdge, edge * 0.55);
  // Territory: an ink stain with frayed edges and a darker rim.
  vec4 terr = texture2D(uTerritory, xz / uTerritorySize);
  float v = terr.r * 2.0 - 1.0;
  float mask = smoothstep(0.15, 0.55, terr.g + (grain - 0.5) * 0.5) * inMap;
  float ink = abs(v) + (grain - 0.5) * 0.55;
  float iaa = max(fwidth(ink), 0.01);
  float own = smoothstep(0.42 - iaa, 0.42 + iaa, ink) * mask;
  float rim = own * (1.0 - smoothstep(0.42, 0.58, ink));
  vec3 inkColor = v > 0.0 ? uInkBlue : uInkRed;
  // Luminance-preserving tint: grass under red ink turns pink, not brown.
  float luma = dot(c, vec3(0.2126, 0.7152, 0.0722));
  float inkLuma = dot(inkColor, vec3(0.2126, 0.7152, 0.0722));
  vec3 tinted = inkColor * (luma / max(inkLuma, 1e-3)) * 1.04;
  c = mix(c, tinted, own * 0.52 * (1.0 - wet));
  c = mix(c, inkColor * 0.72, rim * 0.45 * (1.0 - wet));
  return c;
}`).replace(`#include <color_fragment>`,`#include <color_fragment>
diffuseColor.rgb = groundColor(vGround);`)};let y=new v(o,_);return y.receiveShadow=!0,y.castShadow=!0,y.name=`ground`,{mesh:y,dispose(){o.dispose(),_.dispose(),l.dispose(),u.dispose(),d.dispose()}}}var ji=-.35,Mi=1.6,Ni=6.5,Pi=.9,Fi=.95,Ii=(e,t,n)=>{let r=Math.min(1,Math.max(0,(n-e)/(t-e)));return r*r*(3-2*r)},Li=e=>e.kind===`rect`?{x0:e.x0,y0:e.y0,x1:e.x1,y1:e.y1}:null,Ri=(e,t)=>Math.abs(e.x0-t.x0)<1e-6&&Math.abs(e.x1-t.x1)<1e-6&&Math.abs(e.y0-t.y0)<1e-6&&Math.abs(e.y1-t.y1)<1e-6;function zi(e,t,n){return Ii(e.y0-n,e.y0+n*.25,t)*(1-Ii(e.y1-n*.25,e.y1+n,t))}function Bi(e,t=70){let n=e.water.map(Li).filter(e=>e!==null),r=e.zones.filter(e=>e.kind===`ford`).map(e=>Li(e.shape)).filter(e=>e!==null),i=e.crossings.map(Li).filter(e=>e!==null),a=e=>r.some(t=>Ri(t,e)),o=e.obstacles.filter(e=>e.kind===`circle`),s=e=>o.some(t=>t.y>e.y0-t.r-2&&t.y<e.y1+t.r+2&&t.x>=e.x0&&t.x<=e.x1),c=i.filter(a),l=i.filter(e=>!a(e)&&s(e)),u=i.filter(e=>!a(e)&&!s(e)),d=o.map(t=>{let n=e.zones.find(e=>e.kind===`highGround`&&e.shape.kind===`circle`&&Math.abs(e.shape.x-t.x)<1e-6&&Math.abs(e.shape.y-t.y)<1e-6),r=n&&n.shape.kind===`circle`?n.shape.r:t.r*2;return{x:t.x,y:t.y,r:t.r,skirt:r}}),f={x0:-t,y0:-t,x1:e.width+t,y1:e.height+t},p=(e,t)=>{let n=(e.x0+e.x1)/2,r=(e.x1-e.x0)/2;return Ii(r+.6,r-1.6,Math.abs(t-n))},m=(e,t)=>{let i=0,a=0;for(let s of n){let n=p(s,e);if(n<=0)continue;let c=n*(t<s.y0?1:zi(s,t,1.5));for(let e of l)c*=1-zi(e,t,2.2);for(let n of o)c*=Ii(n.r,n.r+3,Math.hypot(e-n.x,t-n.y));for(let e of r)a=Math.max(a,zi(e,t,3));i=Math.max(i,c)}return{dig:i,ford:a}},h=(e,t,n)=>{let r=0;for(let i of d){let a=Math.hypot(e-i.x,t-i.y);r+=Ni*Ii(i.r+4,i.r*.25,a),r+=Pi*Ii(i.skirt,i.r,a)*(1-n)}return r},g=(e,t)=>{let{dig:n,ford:r}=m(e,t),i=Mi+-.9800000000000001*r;return h(e,t,n)-i*n},_=(e,t)=>{for(let n of u){let r=(n.x0+n.x1)/2,i=(n.x1-n.x0)/2+3;if(t<n.y0-.8||t>n.y1+.8||Math.abs(e-r)>i)continue;let a=Math.abs(e-r)/i;return Fi*(1-a*a)+.05}return null};return{map:e,bounds:f,groundHeight:g,walkHeight:(e,t)=>{let n=_(e,t),r=g(e,t);return n===null?r:Math.max(n,r)},waterDepth:(e,t)=>Math.max(0,ji-g(e,t)),bridges:u,fords:c,hills:d}}var Vi=class{x0;y0;x1;y1;step;cols;rows;data;constructor(e,t,n,r,i,a){this.x0=e,this.y0=t,this.x1=n,this.y1=r,this.step=i,this.cols=Math.round((n-e)/i)+1,this.rows=Math.round((r-t)/i)+1,this.data=new Float32Array(this.cols*this.rows);for(let n=0;n<this.rows;n++)for(let r=0;r<this.cols;r++)this.data[n*this.cols+r]=a(e+r*i,t+n*i)}sample(e,t){let n=(e-this.x0)/this.step,r=(t-this.y0)/this.step,i=this.cols-1,a=this.rows-1;n=n<0?0:n>i?i:n,r=r<0?0:r>a?a:r;let o=Math.min(i-1,Math.floor(n)),s=Math.min(a-1,Math.floor(r)),c=n-o,l=r-s,u=this.data,d=s*this.cols+o,f=u[d]+(u[d+1]-u[d])*c;return f+(u[d+this.cols]+(u[d+this.cols+1]-u[d+this.cols])*c-f)*l}};function Hi(e){let t=new G;for(let n of e.bridges){let r=(n.x0+n.x1)/2,i=(n.x1-n.x0)/2+3,a=(n.y0+n.y1)/2,o=t=>e.walkHeight(t,a),s=new oe;s.moveTo(r-i,-1.4);for(let e=0;e<=24;e++){let t=r-i+2*i*e/24;s.lineTo(t,o(t))}s.lineTo(r+i,-1.4);let c=(n.x1-n.x0)/2-.6;for(let e=0;e<=12;e++){let t=e/12*Math.PI;s.lineTo(r+Math.cos(t)*c,-1.4+Math.sin(t)*1.25)}s.lineTo(r-i,-1.4);let l=n.y1-n.y0;t.add(new _(s,{depth:l,bevelEnabled:!1,curveSegments:1}),H.stone,new E().makeTranslation(0,0,n.y0));let u=new oe;u.moveTo(r-i+.4,o(r-i+.4)-.2);for(let e=0;e<=24;e++){let t=r-i+.4+(2*i-.8)*e/24;u.lineTo(t,o(t)+.55)}for(let e=24;e>=0;e--){let t=r-i+.4+(2*i-.8)*e/24;u.lineTo(t,o(t)-.2)}for(let e of[n.y0-.35,n.y1])t.add(new _(u,{depth:.35,bevelEnabled:!1,curveSegments:1}),H.stoneDark,new E().makeTranslation(0,0,e));for(let e of[r-i+.4,r+i-.4])for(let r of[n.y0-.18,n.y1+.18]){let n=o(e);t.add(new N(.6,1.1,.6),H.stoneDark,K(e,n+.35,r)),t.add(new D(.42,.4,4),H.stone,K(e,n+1.1,r,Math.PI/4))}}return t}function Ui(e,t,n){for(let r of e.fords){for(let n of[r.y0+.5,r.y1-.5])for(let i of[r.x0-1.2,r.x1+1.2]){let r=e.groundHeight(i,n);t.add(new O(.16,.2,1.8,6),H.woodDark,K(i,r+.7,n)),t.add(new N(.7,.12,.14),H.wood,K(i,r+1.45,n,.3))}for(let e=0;e<26;e++){let i=r.x0+.5+n()*(r.x1-r.x0-1),a=n()<.5?r.y0+n()*3:r.y1-n()*3,o=e%3==0?r.y0+n()*(r.y1-r.y0):a,s=.25+n()*.45;t.add(new u(s,0),n()<.5?H.rock:H.rockDark,K(i,ji-s*.3,o,n()*6,1,.45,1))}}}function Wi(e,t,n){for(let r of e.hills)for(let i=0;i<7;i++){let a=n()*Math.PI*2,o=i===0?0:(.2+n()*.4)*r.r,s=r.x+Math.cos(a)*o,c=r.y+Math.sin(a)*o,l=(i===0?1.9:.7+n()*.9)*(1-o/r.r*.3);t.add(new b(l,0),n()<.5?H.rock:H.rockDark,K(s,e.groundHeight(s,c)+l*.35,c,n()*6,1,.7+n()*.6,1))}}function Gi(){return{round:new G().add(new O(.18,.26,1.6,5),H.trunk,K(0,.8,0)).add(new b(1.25,0),H.leaf,K(0,2.4,0,0,1,1.05,1)).add(new b(.8,0),H.leafLight,K(.45,3.15,.2,.5)),pine:new G().add(new O(.14,.2,1.2,5),H.trunk,K(0,.6,0)).add(new D(1.15,2,7),H.pine,K(0,1.9,0)).add(new D(.8,1.6,7),H.pine,K(0,3,0,.4))}}function Ki(e,t,n=1){let r=new d;r.name=`scenery`;let i=st(1234),a=ot(),s=[a],c=Hi(e);Ui(e,c,i),Wi(e,c,i);let l=new v(c.build(),a);l.castShadow=!0,l.receiveShadow=!0,l.name=`props`,r.add(l);let u=t.map,f=u.routes.map(e=>new o(e.key,e.name,e.points,e.widths)),p=(e,t)=>{let n=1/0;for(let r of f){let i=r.project(e,t);n=Math.min(n,Math.abs(i.offset)-r.widthAt(i.distance)/2)}return n},m=[...u.hq.map(e=>({...e,r:15})),...u.barracks.flat().map(e=>({...e,r:7})),...(u.sites??[[],[]]).flat().map(e=>({x:e.x,y:e.y,r:9}))],h=[],g=3.4,_=e.bounds;for(let t=_.y0+2;t<_.y1-2;t+=g)for(let r=_.x0+2;r<_.x1-2;r+=g){let a=r+(i()-.5)*g*.9,o=t+(i()-.5)*g*.9,s=a>-3&&a<u.width+3&&o>-3&&o<u.height+3,c=Math.max(-a,a-u.width,-o,o-u.height),l=ct(a*.045,o*.045),d;if(d=s?l>.66?.55:l>.58?.12:.006:(c>8?.8:.25+l*.4)*Math.min(1,Math.max(0,n)),i()>d||p(a,o)<(s?5:3))continue;let f=e.groundHeight(a,o);if(f<-.05||e.groundHeight(a+2,o)<-.1||e.groundHeight(a-2,o)<-.1||f>3.2||m.some(e=>Math.hypot(e.x-a,e.y-o)<e.r))continue;let _=e.hills.some(e=>Math.hypot(e.x-a,e.y-o)<e.skirt+3);h.push({x:a,y:o,s:.8+i()*.55,pine:_||i()<.22})}let y=Gi(),b=[16777215,15266520,16183504,14674655];for(let t of[`round`,`pine`]){let n=h.filter(e=>e.pine===(t===`pine`));if(n.length===0)continue;let o=y[t].build(),s=new T(o,a,n.length),c=new j;n.forEach((t,n)=>{let r=e.groundHeight(t.x,t.y);s.setMatrixAt(n,K(t.x,r-.1,t.y,i()*6,t.s,t.s*(.9+i()*.25),t.s)),c.setHex(b[Math.floor(i()*b.length)]),s.setColorAt(n,c)}),s.castShadow=!0,s.receiveShadow=!0,s.name=`trees-${t}`,s.computeBoundingSphere(),r.add(s)}return{group:r,dispose(){r.traverse(e=>{e instanceof v&&e.geometry.dispose()});for(let e of s)e.dispose()}}}var qi=`
float hash12(vec2 p) {
  vec3 p3 = fract(vec3(p.xyx) * 0.1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}
float vnoise(vec2 p) {
  vec2 i = floor(p);
  vec2 f = fract(p);
  vec2 u = f * f * (3.0 - 2.0 * f);
  return mix(mix(hash12(i), hash12(i + vec2(1.0, 0.0)), u.x),
             mix(hash12(i + vec2(0.0, 1.0)), hash12(i + vec2(1.0, 1.0)), u.x), u.y);
}
float fbm(vec2 p) {
  return vnoise(p) * 0.55 + vnoise(p * 2.13 + 7.1) * 0.3 + vnoise(p * 4.7 - 3.3) * 0.15;
}
`,Ji=`
in float aDepth;
out float vDepth;
out vec3 vWorld;
out float vDist;
void main() {
  vDepth = aDepth;
  vec4 world = modelMatrix * vec4(position, 1.0);
  vWorld = world.xyz;
  vec4 mv = viewMatrix * world;
  vDist = -mv.z;
  gl_Position = projectionMatrix * mv;
}
`,Yi=`
${Hn}
${qi}
uniform vec3 uDeep;
uniform vec3 uShallow;
uniform vec3 uFoam;
uniform float uTime;
in float vDepth;
in vec3 vWorld;
in float vDist;
void main() {
  if (vDepth <= 0.0) discard;
  float d = smoothstep(0.0, 1.1, vDepth);
  vec3 c = mix(uShallow, uDeep, d);
  // Ripples: thin bright streaks drifting north, broken by noise.
  vec2 p = vWorld.xz;
  float w = vnoise(vec2(p.x * 0.6, p.y * 0.18 + uTime * 0.35));
  float streak = smoothstep(0.78, 0.84, w) * (1.0 - smoothstep(0.84, 0.92, w));
  c = mix(c, uFoam, streak * 0.35 * d);
  // Foam along the shore.
  float shore = 1.0 - smoothstep(0.02, 0.12 + 0.05 * sin(uTime * 1.3 + p.y * 0.7), vDepth);
  c = mix(c, uFoam, shore * 0.75);
  vec3 lit = lambert(c, vec3(0.0, 1.0, 0.0));
  float alpha = mix(0.45, 0.9, d) + shore * 0.2;
  gl_FragColor = vec4(applyFog(lit, vDist), clamp(alpha, 0.0, 1.0));
  #include <colorspace_fragment>
}
`;function Xi(e,t){let n=e.map.water.filter(e=>e.kind===`rect`);if(n.length===0)return null;let r=1/0,i=-1/0;for(let e of n)e.kind===`rect`&&(r=Math.min(r,e.x0-4),i=Math.max(i,e.x1+4));let a=e.bounds.y0,o=e.bounds.y1,s=i-r,c=o-a,l=new F(s,c,Math.round(s*2),Math.round(c));l.rotateX(-Math.PI/2),l.translate(r+s/2,ji,a+c/2);let u=l.attributes.position,d=new Float32Array(u.count);for(let t=0;t<u.count;t++)d[t]=ji-e.groundHeight(u.getX(t),u.getZ(t));l.setAttribute(`aDepth`,new k(d,1));let f=new P({vertexShader:Ji,fragmentShader:Yi,transparent:!0,depthWrite:!1,uniforms:{...t,uDeep:{value:new j(H.waterDeep)},uShallow:{value:new j(H.waterShallow)},uFoam:{value:new j(H.foam)}}}),p=new v(l,f);return p.renderOrder=2,p.name=`water`,p}var Zi={ArrowLeft:[-1,0],ArrowRight:[1,0],ArrowUp:[0,-1],ArrowDown:[0,1]},Qi=class{client;map;config;element;renderer;scene=new ue;camera;rig;terrain;walk;lights;soldiers;territory;particles;effects;structures;sites;gates;infirmaries;golden;rally;ambience;emberAcc=0;highlight;gateZones;scenery;disposers=[];routes;frameListeners=new Set;pendingEvents=[];keys=new Set;raycaster=new ee;ndc=new c;tmp=new A;resizeObserver;timer=new te;lastSnapshot=null;inputEnabled=!0;drag=null;touches=new Map;pan=null;pinch=null;cameraTouched=!1;overviewFill;qualityProfile;governor;pixelRatio;suspended=!1;lastDrawn=-1/0;zoomAnchor=null;animTime=0;shadowsDirty=!0;width=1;height=1;disposed=!1;celebrated=!1;statsAcc={frames:0,time:0,cpu:0,worst:0};profile={pack:0,effects:0,scene:0,render:0};heightAt=(e,t)=>this.walk.sample(e,t);lastStats;constructor(t,r,i,s,c={}){this.client=r,this.map=i,this.config=s;let l=r.world;this.element=document.createElement(`div`),this.element.className=`rcb-viewport`,Object.assign(this.element.style,{position:`absolute`,inset:`0`,overflow:`hidden`,touchAction:`none`}),t.appendChild(this.element);let u=c.quality??n.high;this.qualityProfile=u;let d=c.overviewFill;this.overviewFill=typeof d==`function`?d:()=>d??{x:.97,y:.97},this.renderer=new e({antialias:c.antialias??u.antialias,powerPreference:`high-performance`,stencil:!1}),this.pixelRatio=c.pixelRatio??Math.min(window.devicePixelRatio||1,u.pixelRatioCap),this.governor=c.adaptive&&c.pixelRatio===void 0?new a(this.pixelRatio,Math.min(this.pixelRatio,u.pixelRatioFloor)):null,this.renderer.setPixelRatio(this.pixelRatio),this.renderer.outputColorSpace=he,this.renderer.shadowMap.enabled=c.shadows??u.shadows,this.renderer.shadowMap.type=3,this.renderer.shadowMap.autoUpdate=!1,this.renderer.setClearColor(H.fog),this.renderer.domElement.style.display=`block`,this.renderer.domElement.style.width=`100%`,this.renderer.domElement.style.height=`100%`,this.element.appendChild(this.renderer.domElement),this.terrain=Bi(i);let f=this.terrain.bounds;this.walk=new Vi(f.x0,f.y0,f.x1,f.y1,.5,this.terrain.walkHeight);let p=(e,t)=>this.walk.sample(e,t);this.routes=i.routes.map(e=>new o(e.key,e.name,e.points,e.widths)),this.camera=new ae(38,1,.3,2e3);let m=new A(i.width/2,0,i.height/2);this.lights=Un(this.scene,m,Math.max(i.width,i.height)*.75);let h=this.lights.uniforms,g=Sr();this.scene.add(g),this.ambience=new $e(this.lights,g,this.renderer,this.element),this.territory=new Ti(l.territory),this.territory.setTarget(r.current.territory),this.territory.update(1);let _=Ai(this.terrain,l,this.territory);this.scene.add(_.mesh),this.disposers.push(()=>_.dispose());let v=Xi(this.terrain,h);v&&(this.scene.add(v),this.disposers.push(()=>{v.geometry.dispose(),v.material.dispose()})),this.scenery=Ki(this.terrain,l,u.scenery),this.scene.add(this.scenery.group),this.structures=new Zt(i,s.hq.radius,p),this.structures.onChange=()=>this.shadowsDirty=!0,this.scene.add(this.structures.group),this.sites=new Vt(i,p),this.sites.onChange=()=>this.shadowsDirty=!0,this.scene.add(this.sites.group),this.gates=new hr(p),this.gates.onChange=()=>this.shadowsDirty=!0,this.scene.add(this.gates.group),this.highlight=new xr(i,p),this.scene.add(this.highlight.group),this.gateZones=new vr(i,p),this.scene.add(this.gateZones.group),this.infirmaries=new sn(p,(e,t)=>this.nearestRoutePoint(e,t),s.infirmary.healPerSecond),this.infirmaries.onChange=()=>this.shadowsDirty=!0,this.scene.add(this.infirmaries.group),this.golden=new Mn(p),this.golden.onChange=()=>this.shadowsDirty=!0,this.scene.add(this.golden.group),this.rally=new Bn(i,p,Math.max(1,s.rally.volley.waves)),this.scene.add(this.rally.group),this.soldiers=new yi(h),this.scene.add(this.soldiers.group),this.particles=ar(h,u.particles);for(let e of this.particles.all)this.scene.add(e.mesh);this.effects=new un(this.particles,{heightAt:p,flashGate:e=>this.gates.flash(e),hitGate:e=>this.gates.hit(e),hitHq:e=>this.structures.hitHq(e),siteArrival:(e,t)=>this.sites.arrive(e,t),shake:(e,t,n)=>{let r=this.rig.current,i=Math.hypot(e-r.x,t-r.y),a=Math.max(25,r.distance*.8);i<a&&this.rig.addShake(n*(1-i/a))},hitInfirmary:e=>this.infirmaries.hit(e),endInfirmary:(e,t)=>this.infirmaries.end(e,t),flashGolden:()=>this.golden.flash(),closeGolden:e=>this.golden.close(e),volleySource:(e,t,n)=>{for(let r of this.client.current.volleys)if(r.side===e&&Math.abs(r.x-t)<.01&&Math.abs(r.y-n)<.01)return{x:r.fromX,y:r.fromY,radius:r.radius};return null},reinforcements:(e,t,n,r)=>this.rally.reinforcements(e,t,n,r),suddenDeath:()=>{this.ambience.begin(this.animTime),this.rig.addShake(.7);for(let e of this.map.hq)for(let t=0;t<40;t++)this.effects.emberAt(e.x+(Math.random()-.5)*14,e.y+(Math.random()-.5)*14,Math.random()*6,1.5)}}),this.effects.hqRadius=s.hq.radius,this.effects.volleyTiming={waves:Math.max(1,s.rally.volley.waves),waveInterval:s.rally.volley.waveInterval},this.structures.brazierEmitter=(e,t,n,r)=>{this.client.isPaused()||this.effects.brazier(e,t,n,r)},this.golden.sparkle=(e,t,n,r)=>{this.client.isPaused()||this.effects.goldSparkle(e,t,n,r)},this.resizeObserver=new ResizeObserver(()=>this.resize()),this.resizeObserver.observe(t),this.resize();let y=this.overview();this.cameraTouched=c.camera!==void 0,this.rig=new Ve({x:c.camera?.x??y.x,y:c.camera?.y??y.y,distance:c.camera?.distance??y.distance},i.width,i.height),this.rig.maxDistance=Math.min(340,Math.max(y.distance*1.25,120)),this.applyCamera(),this.lastStats={fps:0,frameMs:0,worstFrameMs:0,cpuMs:0,drawCalls:0,triangles:0,soldiers:0,lod:!1,camera:{...this.rig.current},breakdown:{pack:0,effects:0,scene:0,render:0}};let b=r.onFrame(e=>{e.events.length>0&&this.pendingEvents.push(e.events)});this.disposers.push(b),this.bindInput(),this.renderer.setAnimationLoop(e=>this.frame(e))}screenToWorld(e,t){let n=this.groundUnder(e,t);return!n||n.x<0||n.y<0||n.x>this.map.width||n.y>this.map.height?null:n}worldToScreen(e,t,n=0){let r=this.tmp.set(e,this.walk.sample(e,t)+n,t).project(this.camera),i=(r.x+1)/2*this.width,a=(1-r.y)/2*this.height;return{x:i,y:a,visible:r.z>-1&&r.z<1&&i>=0&&a>=0&&i<=this.width&&a<=this.height}}setGatePreview(e){if(!e){this.gates.setPreview(null,null);return}let t=null;for(let n of this.routes){let r=n.project(e.x,e.y);(!t||Math.abs(r.offset)<Math.abs(t.offset))&&(t={route:n,distance:r.distance,offset:r.offset})}if(!t)return;let n=t.route.tangentAt(t.distance),r=t.route.widthAt(t.distance)/2+this.config.sim.gateLateralMargin,i=t.route.pointAt(t.distance);this.gates.setPreview({...e,x:i.x,y:i.y},{tx:n.x,ty:n.y,halfWidth:r})}pickBarracks(e,t){let n=this.pointerRay(e,t);if(!n)return null;let r=this.raycaster.intersectObjects(this.structures.pickTargets(),!0);for(let e of r)for(let t=e.object;t;t=t.parent){let e=t.userData.barracks;if(e)return{side:e.side,index:e.index}}let i=this.client.current,a=[0,1].flatMap(e=>i.barracks[e].map(t=>({side:e,index:t.index,x:t.x,y:t.y})));return this.nearestOnScreen(a,n,e,t,1.5,22)}pickSite(e,t){let n=this.pointerRay(e,t);if(!n)return null;let r=this.raycaster.intersectObjects(this.sites.pickTargets(),!1);for(let e of r){let t=e.object.userData.site;if(t)return{side:t.side,index:t.index}}return this.nearestOnScreen(this.sites.openSites(),n,e,t,.5,26)}setSiteHighlights(e){this.sites.setHighlights(e)}setGateZones(e){this.gateZones.set(e)}setInfirmaryPreview(e){this.infirmaries.setPreview(e,this.config.infirmary.bodyRadius,this.config.infirmary.reserve)}setRallyAim(e){this.rally.setAim(e)}setSelectedBarracks(e,t){this.structures.setSelected(e,t)}setHighlightedRoute(e){this.highlight.set(e,1)}focus(e,t,n){this.zoomAnchor=null,this.cameraTouched=!0;let i=n===void 0?this.rig.goal.distance:Re(e,t,n,this.width/this.height,r.width,r.height);this.rig.setGoal({x:e,y:t,distance:i})}setCameraInputEnabled(e){this.inputEnabled=e,e||(this.drag=null,this.pan=null,this.pinch=null,this.keys.clear())}setSuspended(e){e===this.suspended||this.disposed||(this.suspended=e,this.renderer.setAnimationLoop(e?null:e=>this.frame(e)),e||this.timer.reset())}onFrame(e){return this.frameListeners.add(e),()=>this.frameListeners.delete(e)}stats(){return this.lastStats}quality(){return{level:this.qualityProfile.level,pixelRatio:this.pixelRatio,shadows:this.renderer.shadowMap.enabled,adaptive:this.governor!==null}}setCamera(e,t=!0){this.zoomAnchor=null,this.cameraTouched=!0,this.rig.setGoal({...this.rig.goal,...e},t),t&&this.applyCamera()}dispose(){if(!this.disposed){this.disposed=!0,this.renderer.setAnimationLoop(null),this.resizeObserver.disconnect();for(let e of this.disposers)e();this.soldiers.dispose(),this.territory.dispose();for(let e of this.particles.all)e.dispose();this.structures.dispose(),this.sites.dispose(),this.gates.dispose(),this.highlight.dispose(),this.scenery.dispose(),this.gateZones.dispose(),this.infirmaries.dispose(),this.golden.dispose(),this.rally.dispose(),this.ambience.dispose(),this.renderer.dispose(),this.renderer.forceContextLoss(),this.renderer.domElement.remove(),this.element.remove(),this.frameListeners.clear()}}resize(){let e=this.element.parentElement??this.element,t=Math.max(1,e.clientWidth),n=Math.max(1,e.clientHeight);if(t===this.width&&n===this.height||(this.width=t,this.height=n,this.renderer.setSize(t,n,!1),this.camera.aspect=t/n,this.camera.updateProjectionMatrix(),!this.rig))return;let r=this.overview();this.rig.maxDistance=Math.min(340,Math.max(r.distance*1.25,120)),this.cameraTouched?this.rig.goal.distance>this.rig.maxDistance&&this.rig.setGoal({...this.rig.goal,distance:this.rig.maxDistance}):this.rig.setGoal(r,!0),this.applyCamera()}overview(){let e=this.map,t=[...e.hq.flatMap(e=>[{x:e.x-this.config.hq.radius-2,y:e.y},{x:e.x+this.config.hq.radius+2,y:e.y}]),...e.routes.flatMap(e=>e.points)],n=this.overviewFill(),r=this.width/this.height,i=e.width/2,a=e.height/2+2,o=Le(t,i,a,r,38,n.x,n.y);if(!n.dy)return{x:i,y:a,distance:o};let s=Fe(o,r,0,n.dy,{x:i,y:a,h:0});return{x:s.x,y:s.y,distance:o}}setPixelRatio(e){Math.abs(e-this.pixelRatio)<.001||(this.pixelRatio=e,this.renderer.setPixelRatio(e),this.renderer.setSize(this.width,this.height,!1))}applyCamera(){let e=this.rig.current,{eye:t,target:n}=Ae(e,this.walk.sample(e.x,e.y)),r=this.rig.shakeNow();this.camera.position.set(t.x+r.x,t.y+r.y,t.z+r.z),this.camera.lookAt(n.x+r.x*.5,n.y,n.z+r.z*.5),this.camera.updateMatrixWorld()}anchored(e){let t=this.zoomAnchor;return Fe(e,this.width/this.height,t.ndcX,t.ndcY,t,(e,t)=>this.walk.sample(e,t))}nearestRoutePoint(e,t){let n=null;for(let r of this.routes){let i=r.project(e,t);(!n||Math.abs(i.offset)<Math.abs(n.offset))&&(n={route:r,distance:i.distance,offset:i.offset})}return n?n.route.pointAt(n.distance):null}emitEmbers(e,t){let n=this.rig.current,r=Math.max(18,Math.min(140,n.distance*.75)),i=Math.max(1,n.distance/55);for(this.emberAcc+=e*t*(30+r*.5);this.emberAcc>=1;){--this.emberAcc;let e=n.x+(Math.random()-.5)*r*2,t=n.y+(Math.random()-.5)*r*1.3;e<0||t<0||e>this.map.width||t>this.map.height||this.effects.emberAt(e,t,Math.random()*3,i)}}pointerRay(e,t){let n=this.renderer.domElement.getBoundingClientRect();return n.width<=0||n.height<=0?null:(this.ndc.set((e-n.left)/n.width*2-1,-((t-n.top)/n.height)*2+1),this.raycaster.setFromCamera(this.ndc,this.camera),n)}nearestOnScreen(e,t,n,r,i,a){let o=null,s=a;for(let a of e){let e=this.worldToScreen(a.x,a.y,i);if(!e.visible)continue;let c=Math.hypot(t.left+e.x-n,t.top+e.y-r);c<s&&(s=c,o={side:a.side,index:a.index})}return o}groundUnder(e,t){let n=this.renderer.domElement.getBoundingClientRect();this.ndc.set((e-n.left)/n.width*2-1,-((t-n.top)/n.height)*2+1),this.raycaster.setFromCamera(this.ndc,this.camera);let r=this.raycaster.ray.origin,i=this.raycaster.ray.direction,a=0,o=Me(r,i,a);for(let e=0;e<4&&o;e++)a=this.walk.sample(o.x,o.y),o=Me(r,i,a);return o}grab(e,t,n){let r=this.groundUnder(t,n);r&&(this.cameraTouched=!0,this.rig.pan(e.x-r.x,e.y-r.y,!0),this.rig.goal={...this.rig.goal,x:this.rig.current.x,y:this.rig.current.y},this.applyCamera())}startPinch(){let[e,t]=[...this.touches.values()];if(!e||!t)return;let n=(e.x+t.x)/2,r=(e.y+t.y)/2,i=this.groundUnder(n,r);this.pan=null,this.zoomAnchor=null,this.pinch=i?{spread:Math.max(1,Math.hypot(e.x-t.x,e.y-t.y)),distance:this.rig.current.distance,anchor:{x:i.x,y:i.y,h:this.walk.sample(i.x,i.y)}}:null}movePinch(){let e=this.pinch,[t,n]=[...this.touches.values()];if(!e||!t||!n)return;let r=Math.max(1,Math.hypot(t.x-n.x,t.y-n.y)),i=Math.min(this.rig.maxDistance,Math.max(Te,e.distance*e.spread/r)),a=this.renderer.domElement.getBoundingClientRect(),o=((t.x+n.x)/2-a.left)/a.width*2-1,s=-(((t.y+n.y)/2-a.top)/a.height)*2+1,c=Fe(i,this.width/this.height,o,s,e.anchor,(e,t)=>this.walk.sample(e,t));this.cameraTouched=!0,this.rig.setGoal({x:c.x,y:c.y,distance:i},!0),this.applyCamera()}bindInput(){let e=this.element,t=e=>e.preventDefault(),n=t=>{if(this.inputEnabled){if(t.button===2){let n=this.groundUnder(t.clientX,t.clientY);if(!n)return;this.drag={pointerId:t.pointerId,anchor:n},this.zoomAnchor=null,e.setPointerCapture(t.pointerId),t.preventDefault()}else if(t.button===0){if(this.touches.set(t.pointerId,{x:t.clientX,y:t.clientY,x0:t.clientX,y0:t.clientY}),e.hasPointerCapture&&!e.hasPointerCapture(t.pointerId))try{e.setPointerCapture(t.pointerId)}catch{}this.touches.size===2?this.startPinch():this.touches.size>2&&(this.pinch=null)}}},r=e=>{if(this.drag&&e.pointerId===this.drag.pointerId){this.grab(this.drag.anchor,e.clientX,e.clientY);return}let t=this.touches.get(e.pointerId);if(t&&this.inputEnabled){if(t.x=e.clientX,t.y=e.clientY,this.touches.size===2)this.pinch&&this.movePinch();else if(this.touches.size===1){if(!this.pan){if(Math.hypot(t.x-t.x0,t.y-t.y0)<=i(e.pointerType))return;let n=this.groundUnder(t.x0,t.y0);if(!n)return;this.pan={pointerId:e.pointerId,anchor:n},this.zoomAnchor=null}this.pan.pointerId===e.pointerId&&this.grab(this.pan.anchor,t.x,t.y)}}},a=t=>{if(this.drag&&t.pointerId===this.drag.pointerId){this.drag=null,e.hasPointerCapture(t.pointerId)&&e.releasePointerCapture(t.pointerId);return}if(!this.touches.delete(t.pointerId))return;e.hasPointerCapture?.(t.pointerId)&&e.releasePointerCapture(t.pointerId),this.pinch=null,this.pan=null;let n=[...this.touches.entries()][0];if(this.touches.size===1&&n){let[e,t]=n,r=this.groundUnder(t.x,t.y);r&&(this.pan={pointerId:e,anchor:r})}else this.touches.size===2&&this.startPinch()},o=e=>{if(!this.inputEnabled)return;e.preventDefault(),this.cameraTouched=!0;let t=e.deltaMode===1?16:e.deltaMode===2?400:1,n=Math.exp(Math.max(-300,Math.min(300,e.deltaY*t))*.0016),r=this.rig.goal,i=Math.min(this.rig.maxDistance,Math.max(Te,r.distance*n)),a=this.groundUnder(e.clientX,e.clientY);if(!a){this.zoomAnchor=null,this.rig.setGoal({...r,distance:i});return}let o=this.renderer.domElement.getBoundingClientRect();this.zoomAnchor={ndcX:(e.clientX-o.left)/o.width*2-1,ndcY:-((e.clientY-o.top)/o.height)*2+1,x:a.x,y:a.y,h:this.walk.sample(a.x,a.y)};let s=this.anchored(i);this.rig.setGoal({x:s.x,y:s.y,distance:i})},s=()=>{let e=document.activeElement;return!!e&&(e.tagName===`INPUT`||e.tagName===`TEXTAREA`||e.isContentEditable)},c=e=>{this.inputEnabled&&!s()&&e.key in Zi&&(this.keys.add(e.key),e.preventDefault())},l=e=>{this.keys.delete(e.key)},u=()=>this.keys.clear();e.addEventListener(`contextmenu`,t),e.addEventListener(`pointerdown`,n),e.addEventListener(`pointermove`,r),e.addEventListener(`pointerup`,a),e.addEventListener(`pointercancel`,a),e.addEventListener(`lostpointercapture`,a),e.addEventListener(`wheel`,o,{passive:!1}),window.addEventListener(`keydown`,c),window.addEventListener(`keyup`,l),window.addEventListener(`blur`,u),this.disposers.push(()=>{e.removeEventListener(`contextmenu`,t),e.removeEventListener(`pointerdown`,n),e.removeEventListener(`pointermove`,r),e.removeEventListener(`pointerup`,a),e.removeEventListener(`pointercancel`,a),e.removeEventListener(`lostpointercapture`,a),e.removeEventListener(`wheel`,o),window.removeEventListener(`keydown`,c),window.removeEventListener(`keyup`,l),window.removeEventListener(`blur`,u)})}frame(e){let t=this.qualityProfile.maxFps;if(t>0&&e-this.lastDrawn<1e3/t-2)return;this.lastDrawn=e;let n=performance.now();this.timer.update(e);let r=Math.min(.1,this.timer.getDelta()),i=this.client.isPaused();i||(this.animTime+=r);let a=this.animTime;if(this.lights.uniforms.uTime.value=a,this.keys.size>0&&this.inputEnabled){let e=0,t=0;for(let n of this.keys){let r=Zi[n];r&&(e+=r[0],t+=r[1])}let n=this.rig.goal.distance*.9*r;(e!==0||t!==0)&&(this.zoomAnchor=null,this.cameraTouched=!0),this.rig.pan(e*n,t*n,!1)}if(this.rig.update(r),this.zoomAnchor){let e=this.rig.current,t=this.anchored(e.distance);this.rig.current=je({x:t.x,y:t.y,distance:e.distance},this.map.width,this.map.height,this.rig.maxDistance),Math.abs(Math.log(e.distance/this.rig.goal.distance))<.001&&(this.zoomAnchor=null)}this.applyCamera();let o=this.rig.current.distance,s=this.client.current,c=s!==this.lastSnapshot;if(c){this.lastSnapshot=s;let e=performance.now();if(this.soldiers.setSoldiers(s.soldiers,a,this.heightAt),this.territory.setTarget(s.territory),this.profile.pack+=performance.now()-e,s.outcome&&!this.celebrated&&s.outcome.winner>=0){this.celebrated=!0;let e=[this.map.hq[1-s.outcome.winner],{x:this.map.width/2,y:this.map.height/2}];this.effects.beginFrame(a),this.effects.celebrate(s.outcome.winner,e)}}let l=this.client.alpha(e);this.soldiers.update(l,s.dt,o),this.territory.update(r);let u=performance.now();this.effects.beginFrame(a);for(let e of this.pendingEvents)this.effects.handle(e);if(this.pendingEvents.length=0,c&&!i&&(this.effects.chargeTrails(s),s.infirmaries.length>0&&this.effects.healMotes(s.infirmaries,s,this.config.infirmary.healPerSecond)),this.ambience.k>0&&!i&&this.emitEmbers(r,this.ambience.k),!i)for(let e of s.hqs)e.crumbling&&!e.destroyed&&this.effects.crumble(e.side,e.x,e.y,r);let d=performance.now();this.profile.effects+=d-u,this.structures.update(s,a,r),this.structures.updateScale(o),this.sites.update(s,a,r,o),this.gates.update(s.gates,a,r,o),this.infirmaries.update(s.infirmaries,s.dt,a,r,o);let f=s.time-s.dt*(1-l);this.golden.update(s.goldenGate,f,a,r,o),this.rally.update(s.charges,s.volleys,a,r),this.ambience.update(s.suddenDeath!==null,a,r),this.highlight.update(a,r),this.gateZones.update(a,r),this.effects.update(),this.shadowsDirty&&=(this.renderer.shadowMap.needsUpdate=!0,!1);let p=performance.now();this.profile.scene+=p-d,this.renderer.render(this.scene,this.camera),this.profile.render+=performance.now()-p;for(let e of this.frameListeners)e(r);let m=performance.now()-n,h=this.statsAcc;if(h.frames++,h.time+=r,h.cpu+=m,h.worst=Math.max(h.worst,r*1e3),h.time>=1){let e=this.governor?.sample(h.frames/h.time,t>0?t:60);e&&this.setPixelRatio(e);let n=this.renderer.info.render;this.lastStats={fps:h.frames/h.time,frameMs:h.time/h.frames*1e3,worstFrameMs:h.worst,cpuMs:h.cpu/h.frames,drawCalls:n.calls,triangles:n.triangles,soldiers:this.soldiers.count,lod:this.soldiers.usingLod,camera:{...this.rig.current},breakdown:{pack:this.profile.pack/h.frames,effects:this.profile.effects/h.frames,scene:this.profile.scene/h.frames,render:this.profile.render/h.frames}},this.profile={pack:0,effects:0,scene:0,render:0},h.frames=0,h.time=0,h.cpu=0,h.worst=0}}};function $i(e,t,n,r,i={}){return new Qi(e,t,n,r,i)}function ea(e={}){return({container:t,client:n,config:r})=>$i(t,n,n.world.map,r,e)}export{$i as n,ea as t};