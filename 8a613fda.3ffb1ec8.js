(window.webpackJsonp=window.webpackJsonp||[]).push([[106],{163:function(e,t,n){"use strict";n.r(t),n.d(t,"frontMatter",(function(){return i})),n.d(t,"metadata",(function(){return p})),n.d(t,"rightToc",(function(){return c})),n.d(t,"default",(function(){return s}));var r=n(2),a=n(7),o=(n(0),n(239)),i={id:"config_groups",title:"Grouping config files"},p={unversionedId:"tutorials/basic/your_first_app/config_groups",id:"tutorials/basic/your_first_app/config_groups",isDocsHomePage:!1,title:"Grouping config files",description:"Example",source:"@site/docs/tutorials/basic/your_first_app/4_config_groups.md",slug:"/tutorials/basic/your_first_app/config_groups",permalink:"/docs/next/tutorials/basic/your_first_app/config_groups",editUrl:"https://github.com/facebookresearch/hydra/edit/master/website/docs/tutorials/basic/your_first_app/4_config_groups.md",version:"current",lastUpdatedBy:"Jieru Hu",lastUpdatedAt:1604440782,sidebar:"docs",previous:{title:"Using the config object",permalink:"/docs/next/tutorials/basic/your_first_app/using_config"},next:{title:"Selecting defaults for config groups",permalink:"/docs/next/tutorials/basic/your_first_app/defaults"}},c=[{value:"Using config groups",id:"using-config-groups",children:[]},{value:"More advanced usages of config groups",id:"more-advanced-usages-of-config-groups",children:[]}],l={rightToc:c};function s(e){var t=e.components,n=Object(a.a)(e,["components"]);return Object(o.b)("wrapper",Object(r.a)({},l,n,{components:t,mdxType:"MDXLayout"}),Object(o.b)("p",null,Object(o.b)("a",Object(r.a)({parentName:"p"},{href:"https://github.com/facebookresearch/hydra/tree/master/examples/tutorials/basic/your_first_hydra_app/4_config_groups"}),Object(o.b)("img",Object(r.a)({parentName:"a"},{src:"https://img.shields.io/badge/-Example-informational",alt:"Example"})))),Object(o.b)("p",null,"Suppose you want to benchmark your application on each of PostgreSQL and MySQL. To do this, use config groups. "),Object(o.b)("p",null,"A ",Object(o.b)("em",{parentName:"p"},Object(o.b)("strong",{parentName:"em"},"Config Group"))," is a named group with a set of valid options."),Object(o.b)("ul",null,Object(o.b)("li",{parentName:"ul"},"The config options are mutually exclusive. Only one can be selected."),Object(o.b)("li",{parentName:"ul"},"Selecting a non-existent config option generates an error message with the valid options.")),Object(o.b)("p",null,"To create a config group, create a directory. e.g. ",Object(o.b)("inlineCode",{parentName:"p"},"db")," to hold a file for each database configuration option.\nSince we are expecting to have multiple config groups, we will proactively move all the configuration files\ninto a ",Object(o.b)("inlineCode",{parentName:"p"},"conf")," directory."),Object(o.b)("pre",null,Object(o.b)("code",Object(r.a)({parentName:"pre"},{className:"language-text",metastring:'title="Directory layout"',title:'"Directory','layout"':!0}),"\u251c\u2500\u2500 conf\n\u2502\xa0\xa0 \u2514\u2500\u2500 db\n\u2502\xa0\xa0     \u251c\u2500\u2500 mysql.yaml\n\u2502\xa0\xa0     \u2514\u2500\u2500 postgresql.yaml\n\u2514\u2500\u2500 my_app.py\n")),Object(o.b)("pre",null,Object(o.b)("code",Object(r.a)({parentName:"pre"},{className:"language-yaml",metastring:'title="db/mysql.yaml"',title:'"db/mysql.yaml"'}),"# @package _group_\ndriver: mysql\nuser: omry\npassword: secret\n")),Object(o.b)("p",null,"The config group determines the ",Object(o.b)("inlineCode",{parentName:"p"},"package")," of the config content inside the final config object.  "),Object(o.b)("pre",null,Object(o.b)("code",Object(r.a)({parentName:"pre"},{className:"language-yaml",metastring:'title="Interpretation of db/mysql.yaml" {1}',title:'"Interpretation',of:!0,'db/mysql.yaml"':!0,"{1}":!0}),"db:\n  driver: mysql\n  user: omry\n  password: secret \n")),Object(o.b)("p",null,"In Hydra 1.1 ",Object(o.b)("inlineCode",{parentName:"p"},"_group_")," will become the default package.",Object(o.b)("br",{parentName:"p"}),"\n","For now, add ",Object(o.b)("inlineCode",{parentName:"p"},"# @package _group_")," at the top of your config group files.",Object(o.b)("br",{parentName:"p"}),"\n","Learn more about packages directive ",Object(o.b)("a",Object(r.a)({parentName:"p"},{href:"/docs/next/advanced/overriding_packages"}),"here"),". "),Object(o.b)("h3",{id:"using-config-groups"},"Using config groups"),Object(o.b)("p",null,"Since we moved all the configs into the ",Object(o.b)("inlineCode",{parentName:"p"},"conf")," directory, we need to tell Hydra where to find them using the ",Object(o.b)("inlineCode",{parentName:"p"},"config_path")," parameter.\n",Object(o.b)("strong",{parentName:"p"},Object(o.b)("inlineCode",{parentName:"strong"},"config_path")," is a directory relative to ",Object(o.b)("inlineCode",{parentName:"strong"},"my_app.py")),"."),Object(o.b)("pre",null,Object(o.b)("code",Object(r.a)({parentName:"pre"},{className:"language-python",metastring:'title="my_app.py" {1}',title:'"my_app.py"',"{1}":!0}),'@hydra.main(config_path="conf")\ndef my_app(cfg: DictConfig) -> None:\n    print(OmegaConf.to_yaml(cfg))\n')),Object(o.b)("p",null,"Running ",Object(o.b)("inlineCode",{parentName:"p"},"my_app.py")," without requesting a configuration will print an empty config."),Object(o.b)("pre",null,Object(o.b)("code",Object(r.a)({parentName:"pre"},{className:"language-yaml"}),"$ python my_app.py\n{}\n")),Object(o.b)("p",null,"You can append an item a config group to the ",Object(o.b)("inlineCode",{parentName:"p"},"Defaults List"),".",Object(o.b)("br",{parentName:"p"}),"\n","The ",Object(o.b)("inlineCode",{parentName:"p"},"Defaults List")," is described on the next page."),Object(o.b)("pre",null,Object(o.b)("code",Object(r.a)({parentName:"pre"},{className:"language-yaml"}),"$ python my_app.py +db=postgresql\ndb:\n  driver: postgresql\n  pass: drowssap\n  timeout: 10\n  user: postgre_user\n")),Object(o.b)("p",null,"Like before, you can still override individual values in the resulting config:"),Object(o.b)("pre",null,Object(o.b)("code",Object(r.a)({parentName:"pre"},{className:"language-yaml"}),"$ python my_app.py +db=postgresql db.timeout=20\ndb:\n  driver: postgresql\n  pass: drowssap\n  timeout: 20\n  user: postgre_user\n")),Object(o.b)("h3",{id:"more-advanced-usages-of-config-groups"},"More advanced usages of config groups"),Object(o.b)("p",null,"Config groups can be nested. For example the config group ",Object(o.b)("inlineCode",{parentName:"p"},"db/mysql/storage_engine")," can contain ",Object(o.b)("inlineCode",{parentName:"p"},"innodb.yaml")," and ",Object(o.b)("inlineCode",{parentName:"p"},"myisam.yaml"),".\nWhen selecting an option from a nested config group, use ",Object(o.b)("inlineCode",{parentName:"p"},"/"),":"),Object(o.b)("pre",null,Object(o.b)("code",Object(r.a)({parentName:"pre"},{}),"$ python my_app.py +db=mysql +db/mysql/storage_engine=innodb\ndb:\n  driver: mysql\n  user: omry\n  password: secret \n  mysql:\n    storage_engine:\n      innodb_data_file_path: /var/lib/mysql/ibdata1\n      max_file_size: 1G\n")),Object(o.b)("p",null,"This simple example also demonstrated a very powerful feature of Hydra:\nYou can compose your configuration object from multiple configuration groups."))}s.isMDXComponent=!0},239:function(e,t,n){"use strict";n.d(t,"a",(function(){return b})),n.d(t,"b",(function(){return m}));var r=n(0),a=n.n(r);function o(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function i(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);t&&(r=r.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,r)}return n}function p(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?i(Object(n),!0).forEach((function(t){o(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):i(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function c(e,t){if(null==e)return{};var n,r,a=function(e,t){if(null==e)return{};var n,r,a={},o=Object.keys(e);for(r=0;r<o.length;r++)n=o[r],t.indexOf(n)>=0||(a[n]=e[n]);return a}(e,t);if(Object.getOwnPropertySymbols){var o=Object.getOwnPropertySymbols(e);for(r=0;r<o.length;r++)n=o[r],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(a[n]=e[n])}return a}var l=a.a.createContext({}),s=function(e){var t=a.a.useContext(l),n=t;return e&&(n="function"==typeof e?e(t):p(p({},t),e)),n},b=function(e){var t=s(e.components);return a.a.createElement(l.Provider,{value:t},e.children)},u={inlineCode:"code",wrapper:function(e){var t=e.children;return a.a.createElement(a.a.Fragment,{},t)}},g=a.a.forwardRef((function(e,t){var n=e.components,r=e.mdxType,o=e.originalType,i=e.parentName,l=c(e,["components","mdxType","originalType","parentName"]),b=s(n),g=r,m=b["".concat(i,".").concat(g)]||b[g]||u[g]||o;return n?a.a.createElement(m,p(p({ref:t},l),{},{components:n})):a.a.createElement(m,p({ref:t},l))}));function m(e,t){var n=arguments,r=t&&t.mdxType;if("string"==typeof e||r){var o=n.length,i=new Array(o);i[0]=g;var p={};for(var c in t)hasOwnProperty.call(t,c)&&(p[c]=t[c]);p.originalType=e,p.mdxType="string"==typeof e?e:r,i[1]=p;for(var l=2;l<o;l++)i[l]=n[l];return a.a.createElement.apply(null,i)}return a.a.createElement.apply(null,n)}g.displayName="MDXCreateElement"}}]);