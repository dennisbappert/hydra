(self.webpackChunkwebsite=self.webpackChunkwebsite||[]).push([[5345],{3905:function(e,t,n){"use strict";n.d(t,{Zo:function(){return d},kt:function(){return f}});var a=n(7294);function i(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function o(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);t&&(a=a.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,a)}return n}function r(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?o(Object(n),!0).forEach((function(t){i(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):o(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function l(e,t){if(null==e)return{};var n,a,i=function(e,t){if(null==e)return{};var n,a,i={},o=Object.keys(e);for(a=0;a<o.length;a++)n=o[a],t.indexOf(n)>=0||(i[n]=e[n]);return i}(e,t);if(Object.getOwnPropertySymbols){var o=Object.getOwnPropertySymbols(e);for(a=0;a<o.length;a++)n=o[a],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(i[n]=e[n])}return i}var s=a.createContext({}),c=function(e){var t=a.useContext(s),n=t;return e&&(n="function"==typeof e?e(t):r(r({},t),e)),n},d=function(e){var t=c(e.components);return a.createElement(s.Provider,{value:t},e.children)},u={inlineCode:"code",wrapper:function(e){var t=e.children;return a.createElement(a.Fragment,{},t)}},p=a.forwardRef((function(e,t){var n=e.components,i=e.mdxType,o=e.originalType,s=e.parentName,d=l(e,["components","mdxType","originalType","parentName"]),p=c(n),f=i,m=p["".concat(s,".").concat(f)]||p[f]||u[f]||o;return n?a.createElement(m,r(r({ref:t},d),{},{components:n})):a.createElement(m,r({ref:t},d))}));function f(e,t){var n=arguments,i=t&&t.mdxType;if("string"==typeof e||i){var o=n.length,r=new Array(o);r[0]=p;var l={};for(var s in t)hasOwnProperty.call(t,s)&&(l[s]=t[s]);l.originalType=e,l.mdxType="string"==typeof e?e:i,r[1]=l;for(var c=2;c<o;c++)r[c]=n[c];return a.createElement.apply(null,r)}return a.createElement.apply(null,n)}p.displayName="MDXCreateElement"},1640:function(e,t,n){"use strict";n.r(t),n.d(t,{frontMatter:function(){return l},contentTitle:function(){return s},metadata:function(){return c},toc:function(){return d},default:function(){return p}});var a=n(2122),i=n(9756),o=(n(7294),n(3905)),r=["components"],l={id:"specializing_config",title:"Specializing configuration",sidebar_label:"Specializing configuration"},s=void 0,c={unversionedId:"patterns/specializing_config",id:"version-0.11/patterns/specializing_config",isDocsHomePage:!1,title:"Specializing configuration",description:"In some cases the desired configuration should depend on other configuration choices.",source:"@site/versioned_docs/version-0.11/patterns/specializing_config.md",sourceDirName:"patterns",slug:"/patterns/specializing_config",permalink:"/docs/0.11/patterns/specializing_config",editUrl:"https://github.com/facebookresearch/hydra/edit/master/website/versioned_docs/version-0.11/patterns/specializing_config.md",version:"0.11",lastUpdatedBy:"Jasha10",lastUpdatedAt:1626840829,formattedLastUpdatedAt:"7/21/2021",frontMatter:{id:"specializing_config",title:"Specializing configuration",sidebar_label:"Specializing configuration"},sidebar:"version-0.11/docs",previous:{title:"Creating objects",permalink:"/docs/0.11/patterns/objects"},next:{title:"Introduction",permalink:"/docs/0.11/configure_hydra/intro"}},d=[{value:"initial config.yaml",id:"initial-configyaml",children:[]},{value:"modified config.yaml",id:"modified-configyaml",children:[]},{value:"dataset_model/cifar10_alexnet.yaml",id:"dataset_modelcifar10_alexnetyaml",children:[]}],u={toc:d};function p(e){var t=e.components,n=(0,i.Z)(e,r);return(0,o.kt)("wrapper",(0,a.Z)({},u,n,{components:t,mdxType:"MDXLayout"}),(0,o.kt)("p",null,"In some cases the desired configuration should depend on other configuration choices.\nFor example, You may want to use only 5 layers in your Alexnet model if the dataset of choice is cifar10, and the default 7 otherwise."),(0,o.kt)("p",null,"We can start with a config that looks like this:"),(0,o.kt)("h3",{id:"initial-configyaml"},"initial config.yaml"),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-yaml"},"defaults:\n  - dataset: imagenet\n  - model: alexnet\n")),(0,o.kt)("p",null,"We want to specialize the config based on the choice of the selected dataset and model:\nFurthermore, we only want to do it for cifar10 and alexnet and not for 3 other combinations."),(0,o.kt)("p",null,"OmegaConf supports value interpolation, we can construct a value that would - at runtime - be a function of other values.\nThe idea is that we can add another element to the defaults list that would load a file name that depends on those two values:"),(0,o.kt)("h3",{id:"modified-configyaml"},"modified config.yaml"),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-yaml"},"defaults:\n  - dataset: imagenet\n  - model: alexnet\n  - dataset_model: ${defaults.0.dataset}_${defaults.1.model}\n    optional: true\n")),(0,o.kt)("p",null,"Let's break this down:"),(0,o.kt)("h4",{id:"dataset_model"},"dataset_model"),(0,o.kt)("p",null,"The key ",(0,o.kt)("inlineCode",{parentName:"p"},"dataset_model")," is an arbitrary directory, it can be anything unique that makes sense, including nested directory like ",(0,o.kt)("inlineCode",{parentName:"p"},"dataset/model"),"."),(0,o.kt)("h4",{id:"defaults0dataset_defaults1model"},"${defaults.0.dataset}_${defaults.1.model}"),(0,o.kt)("p",null,"the value ",(0,o.kt)("inlineCode",{parentName:"p"},"${defaults.0.dataset}_${defaults.1.model}")," is using OmegaConf's ",(0,o.kt)("a",{parentName:"p",href:"https://omegaconf.readthedocs.io/en/latest/usage.html#variable-interpolation"},"variable interpolation"),".\nAt runtime, that value would resolve to ",(0,o.kt)("em",{parentName:"p"},"imagenet_alexnet"),", or ",(0,o.kt)("em",{parentName:"p"},"cifar_resnet")," - depending on the values of defaults.dataset and defaults.model.\nThis a bit clunky because defaults contains a list (I hope to improve this in the future)"),(0,o.kt)("h4",{id:"optional-true"},"optional: true"),(0,o.kt)("p",null,"By default, Hydra would fail with an error if a config specified in the defaults does not exist.\nin this case we only want to specialize cifar10 + alexnet, not all 4 combinations.\nindication optional: true here tells Hydra to just continue if it can't find this file."),(0,o.kt)("p",null,"When specializing config, you usually want to only specify what's different, and not the whole thing.\nWe want the model for alexnet, when trained on cifar - to have 5 layers."),(0,o.kt)("h3",{id:"dataset_modelcifar10_alexnetyaml"},"dataset_model/cifar10_alexnet.yaml"),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-yaml"},"model:\n  num_layers: 5\n")),(0,o.kt)("p",null,"Let's check. Running with the default uses imagenet, so we don't get the specialized version of:"),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-yaml"},"$ python example.py \ndataset:\n  name: imagenet\n  path: /datasets/imagenet\nmodel:\n  num_layers: 7\n  type: alexnet\n")),(0,o.kt)("p",null,"Running with cifar10 dataset, we do get 5 for num_layers:"),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-yaml"},"$ python example.py dataset=cifar10\ndataset:\n  name: cifar10\n  path: /datasets/cifar10\nmodel:\n  num_layers: 5\n  type: alexnet\n")))}p.isMDXComponent=!0}}]);