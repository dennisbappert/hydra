(self.webpackChunkwebsite=self.webpackChunkwebsite||[]).push([[6205],{3905:function(e,t,n){"use strict";n.d(t,{Zo:function(){return p},kt:function(){return g}});var r=n(7294);function o(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function a(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);t&&(r=r.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,r)}return n}function i(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?a(Object(n),!0).forEach((function(t){o(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):a(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function s(e,t){if(null==e)return{};var n,r,o=function(e,t){if(null==e)return{};var n,r,o={},a=Object.keys(e);for(r=0;r<a.length;r++)n=a[r],t.indexOf(n)>=0||(o[n]=e[n]);return o}(e,t);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);for(r=0;r<a.length;r++)n=a[r],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(o[n]=e[n])}return o}var c=r.createContext({}),l=function(e){var t=r.useContext(c),n=t;return e&&(n="function"==typeof e?e(t):i(i({},t),e)),n},p=function(e){var t=l(e.components);return r.createElement(c.Provider,{value:t},e.children)},m={inlineCode:"code",wrapper:function(e){var t=e.children;return r.createElement(r.Fragment,{},t)}},u=r.forwardRef((function(e,t){var n=e.components,o=e.mdxType,a=e.originalType,c=e.parentName,p=s(e,["components","mdxType","originalType","parentName"]),u=l(n),g=o,d=u["".concat(c,".").concat(g)]||u[g]||m[g]||a;return n?r.createElement(d,i(i({ref:t},p),{},{components:n})):r.createElement(d,i({ref:t},p))}));function g(e,t){var n=arguments,o=t&&t.mdxType;if("string"==typeof e||o){var a=n.length,i=new Array(a);i[0]=u;var s={};for(var c in t)hasOwnProperty.call(t,c)&&(s[c]=t[c]);s.originalType=e,s.mdxType="string"==typeof e?e:o,i[1]=s;for(var l=2;l<a;l++)i[l]=n[l];return r.createElement.apply(null,i)}return r.createElement.apply(null,n)}u.displayName="MDXCreateElement"},3899:function(e,t,n){"use strict";n.d(t,{Z:function(){return c},T:function(){return l}});var r=n(2122),o=n(7294),a=n(6742),i=n(2263),s=n(907);function c(e){return o.createElement(a.Z,(0,r.Z)({},e,{to:(t=e.to,c=(0,s.zu)(),(0,i.default)().siteConfig.customFields.githubLinkVersionToBaseUrl[null!=(n=null==c?void 0:c.name)?n:"current"]+t),target:"_blank"}));var t,n,c}function l(e){var t,n=null!=(t=e.text)?t:"Example";return o.createElement(c,e,o.createElement("span",null,"\xa0"),o.createElement("img",{src:"https://img.shields.io/badge/-"+n+"-informational",alt:"Example"}))}},4273:function(e,t,n){"use strict";n.r(t),n.d(t,{frontMatter:function(){return c},contentTitle:function(){return l},metadata:function(){return p},toc:function(){return m},default:function(){return g}});var r=n(2122),o=n(9756),a=(n(7294),n(3905)),i=n(3899),s=["components"],c={id:"dynamic_schema",title:"Dynamic schema with many configs"},l=void 0,p={unversionedId:"tutorials/structured_config/dynamic_schema",id:"version-1.0/tutorials/structured_config/dynamic_schema",isDocsHomePage:!1,title:"Dynamic schema with many configs",description:"In this page we will see how to get runtime type safety for configs with dynamic schema.",source:"@site/versioned_docs/version-1.0/tutorials/structured_config/7_dynamic_schema_many_configs.md",sourceDirName:"tutorials/structured_config",slug:"/tutorials/structured_config/dynamic_schema",permalink:"/docs/1.0/tutorials/structured_config/dynamic_schema",editUrl:"https://github.com/facebookresearch/hydra/edit/master/website/versioned_docs/version-1.0/tutorials/structured_config/7_dynamic_schema_many_configs.md",version:"1.0",lastUpdatedBy:"Jasha10",lastUpdatedAt:1626564323,formattedLastUpdatedAt:"7/17/2021",sidebarPosition:7,frontMatter:{id:"dynamic_schema",title:"Dynamic schema with many configs"},sidebar:"version-1.0/docs",previous:{title:"Static schema with many configs",permalink:"/docs/1.0/tutorials/structured_config/static_schema"},next:{title:"Config Store API",permalink:"/docs/1.0/tutorials/structured_config/config_store"}},m=[],u={toc:m};function g(e){var t=e.components,n=(0,o.Z)(e,s);return(0,a.kt)("wrapper",(0,r.Z)({},u,n,{components:t,mdxType:"MDXLayout"}),(0,a.kt)(i.T,{to:"examples/tutorials/structured_configs/7_dynamic_schema_many_configs",mdxType:"ExampleGithubLink"}),(0,a.kt)("p",null,"In this page we will see how to get runtime type safety for configs with dynamic schema.\nOur top level config contains a single field - ",(0,a.kt)("inlineCode",{parentName:"p"},"db"),", with the type ",(0,a.kt)("inlineCode",{parentName:"p"},"DBConfig"),".\nBased on user choice, we would like its type to be either ",(0,a.kt)("inlineCode",{parentName:"p"},"MySQLConfig")," or ",(0,a.kt)("inlineCode",{parentName:"p"},"PostGreSQLConfig")," at runtime.\nThe two schemas differs: config files that are appropriate for one are inappropriate for the other."),(0,a.kt)("p",null,"For each of the two schemas, we have two options - one for prod and one for staging:"),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-text",metastring:'title="Config directory"',title:'"Config','directory"':!0},"\u251c\u2500\u2500 config.yaml\n\u2514\u2500\u2500 db\n    \u251c\u2500\u2500 mysql_prod.yaml\n    \u251c\u2500\u2500 mysql_staging.yaml\n    \u251c\u2500\u2500 postgresql_prod.yaml\n    \u2514\u2500\u2500 postgresql_staging.yaml\n")),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-python",metastring:'title="my_app.py"',title:'"my_app.py"'},'@dataclass\nclass DBConfig:\n    driver: str = MISSING\n    host: str = MISSING\n\n@dataclass\nclass MySQLConfig(DBConfig):\n    driver: str = "mysql"\n    encoding: str = MISSING\n\n@dataclass\nclass PostGreSQLConfig(DBConfig):\n    driver: str = "postgresql"\n    timeout: int = MISSING\n\n@dataclass\nclass Config:\n    db: DBConfig = MISSING\n\ncs = ConfigStore.instance()\ncs.store(group="schema/db", name="mysql", node=MySQLConfig, package="db")\ncs.store(group="schema/db", name="postgresql", node=PostGreSQLConfig, package="db")\ncs.store(name="config", node=Config)\n\n@hydra.main(config_path="conf", config_name="config")\ndef my_app(cfg: Config) -> None:\n    print(OmegaConf.to_yaml(cfg))\n\nif __name__ == "__main__":\n    my_app()\n')),(0,a.kt)("p",null,"When composing the config, we need to select both the schema and the actual config group option.\nThis is what the defaults list looks like:"),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-yaml",metastring:'title="config.yaml"',title:'"config.yaml"'},"defaults:\n  - schema/db: mysql\n  - db: mysql_staging\n")),(0,a.kt)("p",null,"Let's dissect the how we store the schemas into the ",(0,a.kt)("inlineCode",{parentName:"p"},"ConfigStore"),":"),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-python"},'cs.store(group="schema/db", name="mysql", node=MySQLConfig, package="db")\n')),(0,a.kt)("p",null,"There are several notable things here:"),(0,a.kt)("ul",null,(0,a.kt)("li",{parentName:"ul"},"We use the group ",(0,a.kt)("inlineCode",{parentName:"li"},"schema/db")," and not ",(0,a.kt)("inlineCode",{parentName:"li"},"db"),".",(0,a.kt)("br",{parentName:"li"}),"Config Groups are mutually exclusive, only one option can be selected from a Config Group. We want to select both the schema and the config.\nStoring all schemas in subgroups of the config group schema is good practice. This also helps in preventing name collisions."),(0,a.kt)("li",{parentName:"ul"},"We need to specify the package to be ",(0,a.kt)("inlineCode",{parentName:"li"},"db"),".\nBy default, the package for configs stored in the ",(0,a.kt)("inlineCode",{parentName:"li"},"ConfigStore")," is ",(0,a.kt)("inlineCode",{parentName:"li"},"_group_"),". We want to schematize ",(0,a.kt)("inlineCode",{parentName:"li"},"db")," and not ",(0,a.kt)("inlineCode",{parentName:"li"},"schema.db")," in the config so we have to override that. ")),(0,a.kt)("p",null,"By default, we get the mysql staging config:"),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-text",metastring:'title="$ python my_app.py"',title:'"$',python:!0,'my_app.py"':!0},"db:\n  driver: mysql\n  host: mysql001.staging\n  encoding: utf-8\n")),(0,a.kt)("p",null,"We can change both the schema and the config: "),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-text",metastring:'title="$ python my_app.py schema/db=postgresql db=postgresql_prod"',title:'"$',python:!0,"my_app.py":!0,"schema/db":"postgresql",db:'postgresql_prod"'},"db:\n  driver: postgresql\n  host: postgresql01.prod\n  timeout: 10\n")),(0,a.kt)("p",null,"If we try to use a postgresql config without changing the schema as well we will get an error:"),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-text",metastring:'title="$ python my_app.py db=postgresql_prod"',title:'"$',python:!0,"my_app.py":!0,db:'postgresql_prod"'},"Error merging db=postgresql_prod\nKey 'timeout' not in 'MySQLConfig'\n        full_key: db.timeout\n        reference_type=DBConfig\n        object_type=MySQLConfig\n")))}g.isMDXComponent=!0}}]);