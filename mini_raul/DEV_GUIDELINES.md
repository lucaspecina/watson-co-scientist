# Introduccion general:

Lo que quiero es, en este mismo repo, crear una carpeta una version “mini” e iterativa de este mismo proyecto. Quiero de alguna manera destilar informacion de lo que hay en el resto del proyecto pero EMPEZARLO DESDE CERO y con un plan de ir construyendolo de forma progresiva, de a poco. Puedes usar parte del codigo del proyecto original pero no basarse en eso. Ten en cuenta que el proyecto original fue un fracaso y que no funciona bien, por lo que si copias y armas el mini_raul basandote en eso, va a ir por mal camino.

Entonces, como construirlo? Por pasos, de mayor a menor, empezando por las bases, las fundaciones. SE TIENE QUE CONSTRUIR DE FORMA MUY MODULAR, PARA DESPUES IR AGREGANDOLE FEATURES Y HACIENDO CAMBIOS Y QUE NO SEA PROBLEMATICO.

Hay un archivo llamado [ETHOS.md](http://ETHOS.md) donde se explica el objetivo del proyecto. Hay una parte basada en el paper original del cual sale este proyecto y luego hay otra de un posible sistema y a lo ultimo hay una exlpicacion de las funcionalidades que el sistema deberia tener, de acuerdo a mi resumen. Este archivo contiene el objetivo del proyecto final, no lo que tenemos que construir ahora.

Tambien hay otro archivo llamado DEV_LOG.md. Alli tienes que ir haciendo un seguimiento del plan de desarrollo para cada iteracion y lo que se fue haciendo, resolviendo, problemas, comentarios, contexto, etc. Este documento de DEV deberia ser suficiente para volver en cualquier momento, sin memoria de lo hecho y entender todo lo que se fue haciendo, los pasos, iteraciones, pruebas, etc. SIEMPRE ACTUALIZALO!

# Dev guidelines:

We should:

- FOLLOW BEST CODING PRACTICES (well structured, good, modular architecture, reusable things, not too abstract, so on).
- Develop it an iterative way (from simpler to more complex).
- After each "iteration", you should RUN THE SYSTEM FROM SCRATCH to make sure it works correctly.
- After testing it, we should commit in git the changes. You tell me and I will do it manually (and also analyze the changes).

When you add some files for testing or utilities and so on, do it INSIDE particular folders. Do it in an organized way following best practices, not all in the root.

VERY IMPORTANT: YOU HAVE TO USE THE CONDA ENV: “conda activate co_scientist”

Remember, remove all the unnecessary files and folders in the repo. But if you change big things, we should test the system to see that everything is ok.

FOLLOW BEST PRACTICES but always test in the main system that everything is working after major changes!

The models should be run using an AZURE OPENAI service as the default provider. Also include OPENAI, ollama and others as fallbacks. It should be configurable. I already have a .env file with the credentials for all of them.

SIEMPRE SIGUE EL ETHOS.md, los mini_raul/example_desired_system_outputs y actualiza el ACTUALIZA EL DEV_LOG.md periodicamente. Nunca te olvides.