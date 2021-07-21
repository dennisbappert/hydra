(self.webpackChunkwebsite=self.webpackChunkwebsite||[]).push([[6034],{3905:function(e,r,a){"use strict";a.d(r,{Zo:function(){return d},kt:function(){return u}});var t=a(7294);function n(e,r,a){return r in e?Object.defineProperty(e,r,{value:a,enumerable:!0,configurable:!0,writable:!0}):e[r]=a,e}function i(e,r){var a=Object.keys(e);if(Object.getOwnPropertySymbols){var t=Object.getOwnPropertySymbols(e);r&&(t=t.filter((function(r){return Object.getOwnPropertyDescriptor(e,r).enumerable}))),a.push.apply(a,t)}return a}function o(e){for(var r=1;r<arguments.length;r++){var a=null!=arguments[r]?arguments[r]:{};r%2?i(Object(a),!0).forEach((function(r){n(e,r,a[r])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(a)):i(Object(a)).forEach((function(r){Object.defineProperty(e,r,Object.getOwnPropertyDescriptor(a,r))}))}return e}function p(e,r){if(null==e)return{};var a,t,n=function(e,r){if(null==e)return{};var a,t,n={},i=Object.keys(e);for(t=0;t<i.length;t++)a=i[t],r.indexOf(a)>=0||(n[a]=e[a]);return n}(e,r);if(Object.getOwnPropertySymbols){var i=Object.getOwnPropertySymbols(e);for(t=0;t<i.length;t++)a=i[t],r.indexOf(a)>=0||Object.prototype.propertyIsEnumerable.call(e,a)&&(n[a]=e[a])}return n}var l=t.createContext({}),s=function(e){var r=t.useContext(l),a=r;return e&&(a="function"==typeof e?e(r):o(o({},r),e)),a},d=function(e){var r=s(e.components);return t.createElement(l.Provider,{value:r},e.children)},c={inlineCode:"code",wrapper:function(e){var r=e.children;return t.createElement(t.Fragment,{},r)}},m=t.forwardRef((function(e,r){var a=e.components,n=e.mdxType,i=e.originalType,l=e.parentName,d=p(e,["components","mdxType","originalType","parentName"]),m=s(a),u=n,h=m["".concat(l,".").concat(u)]||m[u]||c[u]||i;return a?t.createElement(h,o(o({ref:r},d),{},{components:a})):t.createElement(h,o({ref:r},d))}));function u(e,r){var a=arguments,n=r&&r.mdxType;if("string"==typeof e||n){var i=a.length,o=new Array(i);o[0]=m;var p={};for(var l in r)hasOwnProperty.call(r,l)&&(p[l]=r[l]);p.originalType=e,p.mdxType="string"==typeof e?e:n,o[1]=p;for(var s=2;s<i;s++)o[s]=a[s];return t.createElement.apply(null,o)}return t.createElement.apply(null,a)}m.displayName="MDXCreateElement"},7061:function(e,r,a){"use strict";a.r(r),a.d(r,{frontMatter:function(){return p},contentTitle:function(){return l},metadata:function(){return s},toc:function(){return d},default:function(){return m}});var t=a(2122),n=a(9756),i=(a(7294),a(3905)),o=["components"],p={id:"nevergrad_sweeper",title:"Nevergrad Sweeper plugin",sidebar_label:"Nevergrad Sweeper plugin"},l=void 0,s={unversionedId:"plugins/nevergrad_sweeper",id:"version-1.0/plugins/nevergrad_sweeper",isDocsHomePage:!1,title:"Nevergrad Sweeper plugin",description:"PyPI",source:"@site/versioned_docs/version-1.0/plugins/nevergrad_sweeper.md",sourceDirName:"plugins",slug:"/plugins/nevergrad_sweeper",permalink:"/docs/1.0/plugins/nevergrad_sweeper",editUrl:"https://github.com/facebookresearch/hydra/edit/master/website/versioned_docs/version-1.0/plugins/nevergrad_sweeper.md",version:"1.0",lastUpdatedBy:"Jasha10",lastUpdatedAt:1626840829,formattedLastUpdatedAt:"7/21/2021",frontMatter:{id:"nevergrad_sweeper",title:"Nevergrad Sweeper plugin",sidebar_label:"Nevergrad Sweeper plugin"},sidebar:"version-1.0/docs",previous:{title:"Ax Sweeper plugin",permalink:"/docs/1.0/plugins/ax_sweeper"},next:{title:"Hydra's command line flags",permalink:"/docs/1.0/advanced/hydra-command-line-flags"}},d=[{value:"Installation",id:"installation",children:[]},{value:"Usage",id:"usage",children:[]},{value:"Example of training using Nevergrad hyperparameter search",id:"example-of-training-using-nevergrad-hyperparameter-search",children:[]},{value:"Defining the parameters",id:"defining-the-parameters",children:[{value:"Defining through commandline override",id:"defining-through-commandline-override",children:[]},{value:"Defining through config file",id:"defining-through-config-file",children:[]}]}],c={toc:d};function m(e){var r=e.components,a=(0,n.Z)(e,o);return(0,i.kt)("wrapper",(0,t.Z)({},c,a,{components:r,mdxType:"MDXLayout"}),(0,i.kt)("p",null,(0,i.kt)("a",{parentName:"p",href:"https://pypi.org/project/hydra-nevergrad-sweeper/"},(0,i.kt)("img",{parentName:"a",src:"https://img.shields.io/pypi/v/hydra-nevergrad-sweeper",alt:"PyPI"})),"\n",(0,i.kt)("img",{parentName:"p",src:"https://img.shields.io/pypi/l/hydra-nevergrad-sweeper",alt:"PyPI - License"}),"\n",(0,i.kt)("img",{parentName:"p",src:"https://img.shields.io/pypi/pyversions/hydra-nevergrad-sweeper",alt:"PyPI - Python Version"}),"\n",(0,i.kt)("a",{parentName:"p",href:"https://pypistats.org/packages/hydra-nevergrad-sweeper"},(0,i.kt)("img",{parentName:"a",src:"https://img.shields.io/pypi/dm/hydra-nevergrad-sweeper.svg",alt:"PyPI - Downloads"})),"\n",(0,i.kt)("a",{parentName:"p",href:"https://github.com/facebookresearch/hydra/tree/1.0_branch/plugins/hydra_nevergrad_sweeper/example"},(0,i.kt)("img",{parentName:"a",src:"https://img.shields.io/badge/-Example%20application-informational",alt:"Example application"})),"\n",(0,i.kt)("a",{parentName:"p",href:"https://github.com/facebookresearch/hydra/tree/1.0_branch/plugins/hydra_nevergrad_sweeper"},(0,i.kt)("img",{parentName:"a",src:"https://img.shields.io/badge/-Plugin%20source-informational",alt:"Plugin source"}))),(0,i.kt)("p",null,(0,i.kt)("a",{parentName:"p",href:"https://facebookresearch.github.io/nevergrad/"},"Nevergrad")," is a derivative-free optimization platform providing a library of state-of-the-art algorithms for hyperparameter search. This plugin provides Hydra applications a mechanism to use ",(0,i.kt)("a",{parentName:"p",href:"https://facebookresearch.github.io/nevergrad/"},"Nevergrad")," algorithms to optimize experiment/application parameters."),(0,i.kt)("h3",{id:"installation"},"Installation"),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-commandline"},"pip install hydra-nevergrad-sweeper --upgrade\n")),(0,i.kt)("h3",{id:"usage"},"Usage"),(0,i.kt)("p",null,"Once installed, add ",(0,i.kt)("inlineCode",{parentName:"p"},"hydra/sweeper=nevergrad")," to your command. Alternatively, override ",(0,i.kt)("inlineCode",{parentName:"p"},"hydra/sweeper")," in your config:"),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-yaml"},"defaults:\n  - hydra/sweeper: nevergrad\n")),(0,i.kt)("p",null,"The default configuration is ",(0,i.kt)("a",{parentName:"p",href:"https://github.com/facebookresearch/hydra/blob/master/plugins/hydra_nevergrad_sweeper/hydra_plugins/hydra_nevergrad_sweeper/config.py"},"here"),"."),(0,i.kt)("h2",{id:"example-of-training-using-nevergrad-hyperparameter-search"},"Example of training using Nevergrad hyperparameter search"),(0,i.kt)("p",null,"We include an example of how to use this plugin. The file ",(0,i.kt)("a",{parentName:"p",href:"https://github.com/facebookresearch/hydra/blob/master/plugins/hydra_nevergrad_sweeper/example/my_app.py"},(0,i.kt)("inlineCode",{parentName:"a"},"example/my_app.py"))," implements an example of minimizing a (dummy) function using a mixture of continuous and discrete parameters."),(0,i.kt)("p",null,"You can discover the Nevergrad sweeper parameters with:"),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-yaml",metastring:'title="$ python your_app hydra/sweeper=nevergrad --cfg hydra -p hydra.sweeper"',title:'"$',python:!0,your_app:!0,"hydra/sweeper":"nevergrad","--cfg":!0,hydra:!0,"-p":!0,'hydra.sweeper"':!0},"# @package hydra.sweeper\n_target_: hydra_plugins.hydra_nevergrad_sweeper.core.NevergradSweeper\noptim:\n  optimizer: OnePlusOne\n  budget: 80\n  num_workers: 10\n  noisy: false\n  maximize: false\n  seed: null\nparametrization: {}\nversion: 1\n")),(0,i.kt)("p",null,"The function decorated with ",(0,i.kt)("inlineCode",{parentName:"p"},"@hydra.main()")," returns a float which we want to minimize, the minimum is 0 and reached for:"),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-yaml"},"db: mnist\nlr: 0.12\ndropout: 0.33\nbatch_size=4\n")),(0,i.kt)("p",null,"To run hyperparameter search and look for the best parameters for this function, clone the code and run the following command in the ",(0,i.kt)("inlineCode",{parentName:"p"},"plugins/hydra_nevergrad_sweeper")," directory:"),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-bash"},"python example/my_app.py -m\n")),(0,i.kt)("p",null,"You can also override the search space parametrization:"),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-bash"},"python example/my_app.py --multirun db=mnist,cifar batch_size=4,8,16 \\\n'lr=tag(log, interval(0.001, 1))' 'dropout=interval(0,1)'\n")),(0,i.kt)("p",null,"The initialization of the sweep and the first 5 evaluations (out of 100) look like this:"),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-text"},"[2020-10-08 20:13:53,592][HYDRA] NevergradSweeper(optimizer=OnePlusOne, budget=100, num_workers=10) minimization\n[2020-10-08 20:13:53,593][HYDRA] with parametrization Dict(batch_size=Choice(choices=Tuple(4,8,16),weights=Array{(1,3)}),db=Choice(choices=Tuple(mnist,cifar),weights=Array{(1,2)}),dropout=Scalar{Cl(0,1,b)}[sigma=Log{exp=2.0}],lr=Log{exp=3.162277660168379,Cl(0.001,1,b)}):{'db': 'mnist', 'lr': 0.03162277660168379, 'dropout': 0.5, 'batch_size': 8}\n[2020-10-08 20:13:53,593][HYDRA] Sweep output dir: multirun/2020-10-08/20-13-53\n[2020-10-08 20:13:55,023][HYDRA] Launching 10 jobs locally\n[2020-10-08 20:13:55,023][HYDRA]        #0 : db=mnist lr=0.03162277660168379 dropout=0.5 batch_size=16\n[2020-10-08 20:13:55,217][__main__][INFO] - dummy_training(dropout=0.500, lr=0.032, db=mnist, batch_size=16) = 13.258\n[2020-10-08 20:13:55,218][HYDRA]        #1 : db=cifar lr=0.018178519762066934 dropout=0.5061074452336254 batch_size=4\n[2020-10-08 20:13:55,408][__main__][INFO] - dummy_training(dropout=0.506, lr=0.018, db=cifar, batch_size=4) = 0.278\n[2020-10-08 20:13:55,409][HYDRA]        #2 : db=cifar lr=0.10056825918734161 dropout=0.6399687427725211 batch_size=4\n[2020-10-08 20:13:55,595][__main__][INFO] - dummy_training(dropout=0.640, lr=0.101, db=cifar, batch_size=4) = 0.329\n[2020-10-08 20:13:55,596][HYDRA]        #3 : db=mnist lr=0.06617542958182834 dropout=0.5059497416026679 batch_size=8\n[2020-10-08 20:13:55,812][__main__][INFO] - dummy_training(dropout=0.506, lr=0.066, db=mnist, batch_size=8) = 5.230\n[2020-10-08 20:13:55,813][HYDRA]        #4 : db=mnist lr=0.16717013388679514 dropout=0.6519070394318255 batch_size=4\n...\n[2020-10-08 20:14:27,988][HYDRA] Best parameters: db=cifar lr=0.11961221693764439 dropout=0.37285878409770895 batch_size=4 \n")),(0,i.kt)("p",null,"and the final 2 evaluations look like this:"),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-text"},"[HYDRA]     #8 : db=mnist batch_size=4 lr=0.094 dropout=0.381\n[__main__][INFO] - my_app.py(dropout=0.381, lr=0.094, db=mnist, batch_size=4) = 1.077\n[HYDRA]     #9 : db=mnist batch_size=4 lr=0.094 dropout=0.381\n[__main__][INFO] - my_app.py(dropout=0.381, lr=0.094, db=mnist, batch_size=4) = 1.077\n[HYDRA] Best parameters: db=mnist batch_size=4 lr=0.094 dropout=0.381\n")),(0,i.kt)("p",null,"The run also creates an ",(0,i.kt)("inlineCode",{parentName:"p"},"optimization_results.yaml")," file in your sweep folder with the parameters recommended by the optimizer:"),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-yaml"},"best_evaluated_result: 0.381\n\nbest_evaluated_params:\n  batch_size: 4\n  db: mnist\n  dropout: 0.381\n  lr: 0.094\n\nname: nevergrad\n")),(0,i.kt)("h2",{id:"defining-the-parameters"},"Defining the parameters"),(0,i.kt)("p",null,"The plugin supports two types of parameters: ",(0,i.kt)("a",{parentName:"p",href:"https://facebookresearch.github.io/nevergrad/parametrization_ref.html#nevergrad.p.Choice"},"Choices")," and ",(0,i.kt)("a",{parentName:"p",href:"https://facebookresearch.github.io/nevergrad/parametrization_ref.html#nevergrad.p.Scalar"},"Scalars"),". They can be defined either through config file or commandline override."),(0,i.kt)("h3",{id:"defining-through-commandline-override"},"Defining through commandline override"),(0,i.kt)("p",null,"Hydra provides a override parser that support rich syntax. More documentation can be found in (",(0,i.kt)("a",{parentName:"p",href:"/docs/1.0/advanced/override_grammar/basic"},"OverrideGrammer/Basic"),") and (",(0,i.kt)("a",{parentName:"p",href:"/docs/1.0/advanced/override_grammar/extended"},"OverrideGrammer/Extended"),"). We recommend you go through them first before proceeding with this doc."),(0,i.kt)("h4",{id:"choices"},"Choices"),(0,i.kt)("p",null,"To override a field with choices:"),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-commandline"},"'key=1,5'\n'key=shuffle(range(1, 8))'      \n'key=range(1,5)'\n")),(0,i.kt)("p",null,"You can tag an override with ",(0,i.kt)("inlineCode",{parentName:"p"},"ordered")," to indicate it's a ",(0,i.kt)("a",{parentName:"p",href:"https://facebookresearch.github.io/nevergrad/parametrization_ref.html#nevergrad.p.TransitionChoice"},(0,i.kt)("inlineCode",{parentName:"a"},"TransitionChoice"))),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-commandline"},"`key=tag(ordered, choice(1,2,3))`\n")),(0,i.kt)("h4",{id:"scalar"},"Scalar"),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-commandline"},"`key=interval(1,12)`             # Interval are float by default\n`key=int(interval(1,8))`         # Scalar bounds cast to a int\n`key=tag(log, interval(1,12))`   # call ng.p.Log if tagged with log \n")),(0,i.kt)("h3",{id:"defining-through-config-file"},"Defining through config file"),(0,i.kt)("h4",{id:"choices-1"},"Choices"),(0,i.kt)("p",null,"Choices are defined with a list in a config file."),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-yaml"},"db:\n  - mnist\n  - cifar\n")),(0,i.kt)("h4",{id:"scalars"},"Scalars"),(0,i.kt)("p",null,"Scalars can be defined in config files, with fields:"),(0,i.kt)("ul",null,(0,i.kt)("li",{parentName:"ul"},(0,i.kt)("inlineCode",{parentName:"li"},"init"),": optional initial value"),(0,i.kt)("li",{parentName:"ul"},(0,i.kt)("inlineCode",{parentName:"li"},"lower")," : optional lower bound"),(0,i.kt)("li",{parentName:"ul"},(0,i.kt)("inlineCode",{parentName:"li"},"upper"),": optional upper bound"),(0,i.kt)("li",{parentName:"ul"},(0,i.kt)("inlineCode",{parentName:"li"},"log"),": set to ",(0,i.kt)("inlineCode",{parentName:"li"},"true")," for log distributed values"),(0,i.kt)("li",{parentName:"ul"},(0,i.kt)("inlineCode",{parentName:"li"},"step"),": optional step size for looking for better parameters. In linear mode, this is an additive step; in logarithmic mode it is multiplicative."),(0,i.kt)("li",{parentName:"ul"},(0,i.kt)("inlineCode",{parentName:"li"},"integer"),": set to ",(0,i.kt)("inlineCode",{parentName:"li"},"true")," for integers (favor floats over integers whenever possible)")),(0,i.kt)("p",null,"Providing only ",(0,i.kt)("inlineCode",{parentName:"p"},"lower")," and ",(0,i.kt)("inlineCode",{parentName:"p"},"upper")," bound will set the initial value to the middle of the range and the step to a sixth of the range.\n",(0,i.kt)("strong",{parentName:"p"},"Note"),": unbounded scalars (scalars with no upper and/or lower bounds) can only be defined through a config file."))}m.isMDXComponent=!0}}]);