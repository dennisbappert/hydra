(window.webpackJsonp=window.webpackJsonp||[]).push([[169],{225:function(e,t,r){"use strict";r.r(t),r.d(t,"frontMatter",(function(){return l})),r.d(t,"metadata",(function(){return c})),r.d(t,"rightToc",(function(){return i})),r.d(t,"default",(function(){return s}));var n=r(2),o=r(7),a=(r(0),r(251)),l={id:"colorlog",title:"Colorlog plugin",sidebar_label:"Colorlog plugin"},c={unversionedId:"plugins/colorlog",id:"plugins/colorlog",isDocsHomePage:!1,title:"Colorlog plugin",description:"PyPI",source:"@site/docs/plugins/colorlog.md",slug:"/plugins/colorlog",permalink:"/docs/next/plugins/colorlog",editUrl:"https://github.com/facebookresearch/hydra/edit/master/website/docs/plugins/colorlog.md",version:"current",lastUpdatedBy:"Omry Yadan",lastUpdatedAt:1609539446,sidebar_label:"Colorlog plugin",sidebar:"docs",previous:{title:"Customizing Application's help",permalink:"/docs/next/configure_hydra/app_help"},next:{title:"Joblib Launcher plugin",permalink:"/docs/next/plugins/joblib_launcher"}},i=[{value:"Installation",id:"installation",children:[]},{value:"Usage",id:"usage",children:[]}],p={rightToc:i};function s(e){var t=e.components,l=Object(o.a)(e,["components"]);return Object(a.b)("wrapper",Object(n.a)({},p,l,{components:t,mdxType:"MDXLayout"}),Object(a.b)("p",null,Object(a.b)("a",Object(n.a)({parentName:"p"},{href:"https://pypi.org/project/hydra-colorlog/"}),Object(a.b)("img",Object(n.a)({parentName:"a"},{src:"https://img.shields.io/pypi/v/hydra-colorlog",alt:"PyPI"}))),"\n",Object(a.b)("img",Object(n.a)({parentName:"p"},{src:"https://img.shields.io/pypi/l/hydra-colorlog",alt:"PyPI - License"})),"\n",Object(a.b)("img",Object(n.a)({parentName:"p"},{src:"https://img.shields.io/pypi/pyversions/hydra-colorlog",alt:"PyPI - Python Version"})),"\n",Object(a.b)("a",Object(n.a)({parentName:"p"},{href:"https://pypistats.org/packages/hydra-colorlog"}),Object(a.b)("img",Object(n.a)({parentName:"a"},{src:"https://img.shields.io/pypi/dm/hydra-colorlog.svg",alt:"PyPI - Downloads"}))),"\n",Object(a.b)("a",Object(n.a)({parentName:"p"},{href:"https://github.com/facebookresearch/hydra/tree/master/plugins/hydra_colorlog/example"}),Object(a.b)("img",Object(n.a)({parentName:"a"},{src:"https://img.shields.io/badge/-Example%20application-informational",alt:"Example application"}))),"\n",Object(a.b)("a",Object(n.a)({parentName:"p"},{href:"https://github.com/facebookresearch/hydra/tree/master/plugins/hydra_colorlog"}),Object(a.b)("img",Object(n.a)({parentName:"a"},{src:"https://img.shields.io/badge/-Plugin%20source-informational",alt:"Plugin source"})))),Object(a.b)("p",null,"Adds ",Object(a.b)("a",{class:"external",href:"https://github.com/borntyping/python-colorlog",target:"_blank"},"colorlog")," colored logs for ",Object(a.b)("inlineCode",{parentName:"p"},"hydra/job_logging")," and ",Object(a.b)("inlineCode",{parentName:"p"},"hydra/hydra_logging"),"."),Object(a.b)("h3",{id:"installation"},"Installation"),Object(a.b)("pre",null,Object(a.b)("code",Object(n.a)({parentName:"pre"},{className:"language-commandline"}),"pip install hydra_colorlog --upgrade\n")),Object(a.b)("h3",{id:"usage"},"Usage"),Object(a.b)("p",null,"Override ",Object(a.b)("inlineCode",{parentName:"p"},"hydra/job_logging")," and ",Object(a.b)("inlineCode",{parentName:"p"},"hydra/hydra_logging")," your config:"),Object(a.b)("pre",null,Object(a.b)("code",Object(n.a)({parentName:"pre"},{className:"language-yaml"}),"defaults:\n  - override hydra/job_logging: colorlog\n  - override hydra/hydra_logging: colorlog\n")),Object(a.b)("p",null,"See included ",Object(a.b)("a",Object(n.a)({parentName:"p"},{href:"https://github.com/facebookresearch/hydra/tree/master/plugins/hydra_colorlog/example"}),"example"),"."),Object(a.b)("p",null,Object(a.b)("img",{alt:"Colored log output",src:r(262).default})))}s.isMDXComponent=!0},251:function(e,t,r){"use strict";r.d(t,"a",(function(){return b})),r.d(t,"b",(function(){return d}));var n=r(0),o=r.n(n);function a(e,t,r){return t in e?Object.defineProperty(e,t,{value:r,enumerable:!0,configurable:!0,writable:!0}):e[t]=r,e}function l(e,t){var r=Object.keys(e);if(Object.getOwnPropertySymbols){var n=Object.getOwnPropertySymbols(e);t&&(n=n.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),r.push.apply(r,n)}return r}function c(e){for(var t=1;t<arguments.length;t++){var r=null!=arguments[t]?arguments[t]:{};t%2?l(Object(r),!0).forEach((function(t){a(e,t,r[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(r)):l(Object(r)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(r,t))}))}return e}function i(e,t){if(null==e)return{};var r,n,o=function(e,t){if(null==e)return{};var r,n,o={},a=Object.keys(e);for(n=0;n<a.length;n++)r=a[n],t.indexOf(r)>=0||(o[r]=e[r]);return o}(e,t);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);for(n=0;n<a.length;n++)r=a[n],t.indexOf(r)>=0||Object.prototype.propertyIsEnumerable.call(e,r)&&(o[r]=e[r])}return o}var p=o.a.createContext({}),s=function(e){var t=o.a.useContext(p),r=t;return e&&(r="function"==typeof e?e(t):c(c({},t),e)),r},b=function(e){var t=s(e.components);return o.a.createElement(p.Provider,{value:t},e.children)},g={inlineCode:"code",wrapper:function(e){var t=e.children;return o.a.createElement(o.a.Fragment,{},t)}},u=o.a.forwardRef((function(e,t){var r=e.components,n=e.mdxType,a=e.originalType,l=e.parentName,p=i(e,["components","mdxType","originalType","parentName"]),b=s(r),u=n,d=b["".concat(l,".").concat(u)]||b[u]||g[u]||a;return r?o.a.createElement(d,c(c({ref:t},p),{},{components:r})):o.a.createElement(d,c({ref:t},p))}));function d(e,t){var r=arguments,n=t&&t.mdxType;if("string"==typeof e||n){var a=r.length,l=new Array(a);l[0]=u;var c={};for(var i in t)hasOwnProperty.call(t,i)&&(c[i]=t[i]);c.originalType=e,c.mdxType="string"==typeof e?e:n,l[1]=c;for(var p=2;p<a;p++)l[p]=r[p];return o.a.createElement.apply(null,l)}return o.a.createElement.apply(null,r)}u.displayName="MDXCreateElement"},262:function(e,t,r){"use strict";r.r(t),t.default=r.p+"assets/images/colorlog-b20147697b9d16362f62a5d0bb58347f.png"}}]);