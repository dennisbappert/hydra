(window.webpackJsonp=window.webpackJsonp||[]).push([[42],{247:function(e,t,r){"use strict";r.d(t,"a",(function(){return p})),r.d(t,"b",(function(){return b}));var n=r(0),o=r.n(n);function a(e,t,r){return t in e?Object.defineProperty(e,t,{value:r,enumerable:!0,configurable:!0,writable:!0}):e[t]=r,e}function i(e,t){var r=Object.keys(e);if(Object.getOwnPropertySymbols){var n=Object.getOwnPropertySymbols(e);t&&(n=n.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),r.push.apply(r,n)}return r}function c(e){for(var t=1;t<arguments.length;t++){var r=null!=arguments[t]?arguments[t]:{};t%2?i(Object(r),!0).forEach((function(t){a(e,t,r[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(r)):i(Object(r)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(r,t))}))}return e}function l(e,t){if(null==e)return{};var r,n,o=function(e,t){if(null==e)return{};var r,n,o={},a=Object.keys(e);for(n=0;n<a.length;n++)r=a[n],t.indexOf(r)>=0||(o[r]=e[r]);return o}(e,t);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);for(n=0;n<a.length;n++)r=a[n],t.indexOf(r)>=0||Object.prototype.propertyIsEnumerable.call(e,r)&&(o[r]=e[r])}return o}var u=o.a.createContext({}),s=function(e){var t=o.a.useContext(u),r=t;return e&&(r="function"==typeof e?e(t):c(c({},t),e)),r},p=function(e){var t=s(e.components);return o.a.createElement(u.Provider,{value:t},e.children)},f={inlineCode:"code",wrapper:function(e){var t=e.children;return o.a.createElement(o.a.Fragment,{},t)}},d=o.a.forwardRef((function(e,t){var r=e.components,n=e.mdxType,a=e.originalType,i=e.parentName,u=l(e,["components","mdxType","originalType","parentName"]),p=s(r),d=n,b=p["".concat(i,".").concat(d)]||p[d]||f[d]||a;return r?o.a.createElement(b,c(c({ref:t},u),{},{components:r})):o.a.createElement(b,c({ref:t},u))}));function b(e,t){var r=arguments,n=t&&t.mdxType;if("string"==typeof e||n){var a=r.length,i=new Array(a);i[0]=d;var c={};for(var l in t)hasOwnProperty.call(t,l)&&(c[l]=t[l]);c.originalType=e,c.mdxType="string"==typeof e?e:n,i[1]=c;for(var u=2;u<a;u++)i[u]=r[u];return o.a.createElement.apply(null,i)}return o.a.createElement.apply(null,r)}d.displayName="MDXCreateElement"},98:function(e,t,r){"use strict";r.r(t),r.d(t,"frontMatter",(function(){return i})),r.d(t,"metadata",(function(){return c})),r.d(t,"rightToc",(function(){return l})),r.d(t,"default",(function(){return s}));var n=r(2),o=r(7),a=(r(0),r(247)),i={id:"internal-fb-cluster",title:"Hydra on the internet FB Cluster"},c={unversionedId:"fb/internal-fb-cluster",id:"version-1.0/fb/internal-fb-cluster",isDocsHomePage:!1,title:"Hydra on the internet FB Cluster",description:"Support for launching jobs to the AI cluster is currently still experimental and is expected to evolve over",source:"@site/versioned_docs/version-1.0/fb/ai-cluster.md",slug:"/fb/internal-fb-cluster",permalink:"/docs/fb/internal-fb-cluster",editUrl:"https://github.com/facebookresearch/hydra/edit/master/website/versioned_docs/version-1.0/fb/ai-cluster.md",version:"1.0",lastUpdatedBy:"Jieru Hu",lastUpdatedAt:1608747106},l=[{value:"flow-cli",id:"flow-cli",children:[]},{value:"fry",id:"fry",children:[]}],u={rightToc:l};function s(e){var t=e.components,r=Object(o.a)(e,["components"]);return Object(a.b)("wrapper",Object(n.a)({},u,r,{components:t,mdxType:"MDXLayout"}),Object(a.b)("p",null,"Support for launching jobs to the AI cluster is currently still experimental and is expected to evolve over\nthe coming months."),Object(a.b)("h2",{id:"flow-cli"},"flow-cli"),Object(a.b)("p",null,"flow-cli integration is hacky at the moment.\nSee the the sample f6.sample_projects.classy_hydra_project.workflow.main for details."),Object(a.b)("pre",null,Object(a.b)("code",Object(n.a)({parentName:"pre"},{className:"language-bash",metastring:'title="Example run"',title:'"Example','run"':!0}),'$ CFG=\'{"config": {"overrides": ["trainer=multi_gpu","trainer.max_epochs=90","+lr_scheduler=multi_step"]}}\'\n$ ENTITLEMENT=cv_images_gpu_prod\n$ TEAM=team_computer_vision\n$ WORKFLOW=f6.sample_projects.classy_hydra_project.workflow.main\n$ flow-cli canary $WORKFLOW --run-as-secure-group $TEAM --parameters-json=$CFG --entitlement $ENTITLEMENT\n')),Object(a.b)("h2",{id:"fry"},"fry"),Object(a.b)("p",null,"TODO"))}s.isMDXComponent=!0}}]);