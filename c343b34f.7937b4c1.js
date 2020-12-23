(window.webpackJsonp=window.webpackJsonp||[]).push([[135],{191:function(e,t,r){"use strict";r.r(t),r.d(t,"frontMatter",(function(){return c})),r.d(t,"metadata",(function(){return i})),r.d(t,"rightToc",(function(){return p})),r.d(t,"default",(function(){return u}));var n=r(2),a=r(7),o=(r(0),r(247)),c={id:"search_path",title:"Config search path",sidebar_label:"Config search path"},i={unversionedId:"advanced/search_path",id:"version-0.11/advanced/search_path",isDocsHomePage:!1,title:"Config search path",description:"Hydra uses a search path approach to find configuration files as it composes the configuration object.",source:"@site/versioned_docs/version-0.11/advanced/search_path.md",slug:"/advanced/search_path",permalink:"/docs/0.11/advanced/search_path",editUrl:"https://github.com/facebookresearch/hydra/edit/master/website/versioned_docs/version-0.11/advanced/search_path.md",version:"0.11",lastUpdatedBy:"Jieru Hu",lastUpdatedAt:1608747106,sidebar_label:"Config search path",sidebar:"version-0.11/docs",previous:{title:"Application packaging",permalink:"/docs/0.11/advanced/app_packaging"},next:{title:"Hydra plugins",permalink:"/docs/0.11/advanced/plugins"}},p=[],s={rightToc:p};function u(e){var t=e.components,r=Object(a.a)(e,["components"]);return Object(o.b)("wrapper",Object(n.a)({},s,r,{components:t,mdxType:"MDXLayout"}),Object(o.b)("p",null,"Hydra uses a search path approach to find configuration files as it composes the configuration object.\n",Object(o.b)("inlineCode",{parentName:"p"},"SearchPathPlugin")," can manipulate the search path."),Object(o.b)("p",null,"You can inspect the search path and the configurations loaded by Hydra by turning on verbose logging for the ",Object(o.b)("inlineCode",{parentName:"p"},"hydra")," logger:"),Object(o.b)("pre",null,Object(o.b)("code",Object(n.a)({parentName:"pre"},{className:"language-text"}),"$ python my_app.py hydra.verbose=hydra\n")))}u.isMDXComponent=!0},247:function(e,t,r){"use strict";r.d(t,"a",(function(){return d})),r.d(t,"b",(function(){return h}));var n=r(0),a=r.n(n);function o(e,t,r){return t in e?Object.defineProperty(e,t,{value:r,enumerable:!0,configurable:!0,writable:!0}):e[t]=r,e}function c(e,t){var r=Object.keys(e);if(Object.getOwnPropertySymbols){var n=Object.getOwnPropertySymbols(e);t&&(n=n.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),r.push.apply(r,n)}return r}function i(e){for(var t=1;t<arguments.length;t++){var r=null!=arguments[t]?arguments[t]:{};t%2?c(Object(r),!0).forEach((function(t){o(e,t,r[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(r)):c(Object(r)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(r,t))}))}return e}function p(e,t){if(null==e)return{};var r,n,a=function(e,t){if(null==e)return{};var r,n,a={},o=Object.keys(e);for(n=0;n<o.length;n++)r=o[n],t.indexOf(r)>=0||(a[r]=e[r]);return a}(e,t);if(Object.getOwnPropertySymbols){var o=Object.getOwnPropertySymbols(e);for(n=0;n<o.length;n++)r=o[n],t.indexOf(r)>=0||Object.prototype.propertyIsEnumerable.call(e,r)&&(a[r]=e[r])}return a}var s=a.a.createContext({}),u=function(e){var t=a.a.useContext(s),r=t;return e&&(r="function"==typeof e?e(t):i(i({},t),e)),r},d=function(e){var t=u(e.components);return a.a.createElement(s.Provider,{value:t},e.children)},l={inlineCode:"code",wrapper:function(e){var t=e.children;return a.a.createElement(a.a.Fragment,{},t)}},f=a.a.forwardRef((function(e,t){var r=e.components,n=e.mdxType,o=e.originalType,c=e.parentName,s=p(e,["components","mdxType","originalType","parentName"]),d=u(r),f=n,h=d["".concat(c,".").concat(f)]||d[f]||l[f]||o;return r?a.a.createElement(h,i(i({ref:t},s),{},{components:r})):a.a.createElement(h,i({ref:t},s))}));function h(e,t){var r=arguments,n=t&&t.mdxType;if("string"==typeof e||n){var o=r.length,c=new Array(o);c[0]=f;var i={};for(var p in t)hasOwnProperty.call(t,p)&&(i[p]=t[p]);i.originalType=e,i.mdxType="string"==typeof e?e:n,c[1]=i;for(var s=2;s<o;s++)c[s]=r[s];return a.a.createElement.apply(null,c)}return a.a.createElement.apply(null,r)}f.displayName="MDXCreateElement"}}]);