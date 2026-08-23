
Esta investigación intentar interpretar cómo los jailbreaks basados en ficción rompen los LLMs.
Un ejemplo de este tipo de jailbreaks es: 

En general, estos jailbreaks suelen basarse en insertar una request dañina dentro de algún tipo de ficción y romper así las defensas del modelo, osea el refusal. Pero esto es una afirmación sobre los mecanismos internos del modelo que no fue testeada como tal. 
Entonces la pregunta es: primero, se puede encontrar una dirección lineal que represente qué tan ficcional o narrativo es un texto? En segundo lugar, si existe y suprimo esta dirección, los jailbreaks van a dejar de funciona?
Como ya hay otras direcciones que se ha probado que afectan el refusal, voy a comparar esta nueva dirección de story contra otras 3 para saber si realmente encontramos un mecanismo diferente: 
- primero la más cercana que es persona (qué tanto el modelo hace role-play de un cierto personaje ficticio o qué tanto sólo se comporta como un asistente de AI), 
- después harm (que mide si el modelo detecta si la request es dañina o inofensiva)
- eval-awareness, si sabe que está siendo evaluado, dado que los modelos que saben que están siendo evaluados pueden aumentar las defensas contra jailbreaks

Para poder extraer las direcciones, hice 4 datasets, uno por cada dirección, de 1000 pares de prompts cada uno, en donde la diferencia en cada par representa la dirección target. Por ejemplo, para dirección de story un par puede ser una historia sobre un deportista que alcanza un récord mundial y consigue fama, y el par contrastado es una descripción genérica sobre el deporte.
Entonces, hacemos un diff-of-means para extraer cada dirección: tomamos las activaciones del modelo al darle estos pares y las resto para extraer un vector que corresponde a la dirección que quería. 
Después, para hacer el steering, es decir, para potenciar el comportamiento elegido o para suprimirlo, simplemente añado o resto este vector en las activaciones del modelo.

Entonces, para probar el steering, pasamos 1000 jailbreaks al modelo base y los separamos en aquellos que fueron exitosos de los que no, osea lo que rompieron o no al modelo. 
Para aquellos que fueron exitosos, hacemos el steering para intentar que el modelo se rehúse a responder, y aquellos que no fueron exitosos, hacemos el steering en la dirección opuesta para intentar romper el modelo.

Este grafico en el eje Y mide el attack success rate el ASR, osea si el jailbreak funcionó o no, entre 0 y 100. En la izquierda, tenemos el ASR sobre jailbreaks que ya habían funcionado en un principio, que tenían un ASR de 100, pero ahora al hacer el steering sobre las distintas direcciones, bajó el ASR, haciendo que el modelo se rehúse más seguido. Entonces por ejemplo en gemma, la dirección de historia redujo en un 70% las respuestas dañinas.
Mientras en la derecha tenemos lo contrario: el ASR sobre aquellos jailbreaks que no habían funcionado, es decir que tenian un ASR de 0, pero que ahora después del steering sube su ASR en todas las direcciones, permitiendo romper el modelo. La direccion de historia en este caso en gemma hizo que funcionaran el 30% de los jailbreaks que no habían funcionado. 
Entonces el grafico muestra que las 4 direcciones pueden cambiar qué tanto el modelo se rehúsa a contestar, tanto si potenciamos como si suprimimos las direcciones, aunque también se ve que el efecto es asimétrico: es más fácil que vuelva el refusal a romper el modelo.

Ahora sabiendo que las 4 direcciones tienen un poder causal, la pregunta es: son diferentes?
En el gráfico de la izquierda vemos la cosine similarity entre los vectores, que es una medida entre 0 y 1 de qué tan parecidas son las direcciones geométricamente. Vemos que la medida es baja para todos los pares, osea son diferentes en este sentido. 
Aún así, uno de los pares con más cosine similarity es persona y story, que era lo esperado, dado que los dos tienen componentes ficticios. Por lo tanto, para saber si story y persona son causalmente diferentes hacemos el steering de story sacándole la dirección de persona y vemos que el ASR no cambia, osea que story no necesitaba de la dirección de persona para cambiar el comportamiento del modelo, su efecto propio es suficiente. 
De la misma manera, si hacemos el steering de persona sacándole la dirección de story, el ASR tampoco cambia.

Dos conclusiones principales:
- La dirección de narratividad existe y moverla cambia el comportamiento de los modelos en jailbreaks (por lo que debería ser una dirección más a tener en cuenta y monitorear a la hora de poner en producción modelos)
- La dirección de narratividad es geométricamente y causalmente diferente a las otras 3: tiene un mecanismo distinto para afectar el refusal.