# GPT sentiment

Estrategia:

1. Entrenar un modelo no-supervisado (autoregresivo) para generación de texto
2. Utilizar el modelo para crear una representación de los datos (i.e. embeddings)
3. Agregar una cabeza de clasificación y utilizar la representación sobre el texto del input y las clases

Se puede ver las configuraciones de entrenamiento en W&B: [`sentiment_gpt_esp`](https://wandb.ai/alcazar90/sentiment_gpt_esp?workspace=user-alcazar90).

Algunos textos generados por el modelo: 

```bash
(1)   El Europeo y la fue agresadez y con tiempo de que decir la mujer. Viejeron.@user Aumitivos llegando la pelotude para favor se promuevitieron a la forma que en el duelo se demora en el año play nlamas telesería que era pensar a jajajajajajajajajaja 
(2)                 Era mujer ya bañaja que ni fanas de asqué https://t.co/javs3Tq7D5@user @user Cacholina qliao 😁️jajajaja. Con “fallos van como si no los venecos de comunistas” para que me puede que velgaró que temblaste las ilegalitarias en la venderá
(3)                 #Partifue #PrenséeeeeEsta revuelva lucha el rataje 💩@user Si creerá el arco@user Porque si han una funcho jamás en zorra de casa agua con el articulo del Comunismo chileno culiao biseño corre hará la saron apropinio disidencia de saña
(4)     Llevo es el basurdo sallende su típico. No les cree que así. Feliz Mariaripan es culiao (el tema que tío que @user) y los fachos en el mar tanto, re el mundo a la manos a listas por lo que me encontró con sus comentajes de los horas sido a lo sim
(5)                                                    ""          jsjejeje .tendie plata https://t.co/UT22ckfFl@user ""¿Este prodijo e pita!!! No me ha llegimado un primer retiropidicio malones, realistas de los chilenos.@user @user @user  Migrante gran
(6)         En que la mujer de cuando le desprecieros a todo tanto triste maraca ..!!Le hacía la final que con estos comunistas por favor de @user delincuentes que estaban en la cadera dijo, te viejo lleno de perra@user Siiiiii herense un gusto negro a u
(7)          .  Helloleoooooor, dejé sinflaracer esa wea hincha culiao culiao es un poquitoAJAJAJAJAJA Mexicano él mentiro¡¡ https://t.co/X2meCeApG8@user @user Tremenda que las limpistas medios y es marginal weon con nada bueno mañana pasadita por Orrego
(8)   ???? q https:/.co/d0XSJgIo"@user Va a estas weas on toda maldito @user  callarde!!! Ya correes una feminista te diga en nosotros. Ahi la puta madreee sin contra Naza, jajajajaja o no respetan ser razones. Sabrán su mintro flojo y mis peguitas llena
(9)           de una podriamineros😂😂Si salgan para tantos inicipación re en este mundo y destruyen ctm https://t.co/sqM9PYWvfhApoye la derecha les impitimos con los otros representables. Que deciro que los políticos no tienen una impuesta tienes un artís
(10)    @userEso si no es el limine 🤗"@user Porque es un sincrepto de negro es culiao amiga para quedarse en el starro de la grandesa Inilegal. https://t.co/ilefVfHc7Q@user Esta mierda, viejo culicio para justiciar hasta comunista los otros y pueblistas o
(11) Es ignorante en pelotas"""que le gustaría sacar brella penal"""mañana"En Nunconstrena el cáncejete de Afernuir por otro delincuentes que no hay tomas personas que vienen en Chile existen Pedófilo ctm@user @user @user Más weones venezolanos con muchos
(12)          a que   Q    iga no necesaría q así el fue de hasta q, no me diga la piojera del Cosstall como los weones negros? https://t.co/noQU4HjkoPar qué puedes despues que sufren https://t.co/OJk9j0nfiLpei se creen lo que se viene el barsante con las
(13)   #FaltoCTV@user claro @lsjer y cacho k necni con x las cosas Estado!@user No tienes ni rescalizar como si fuera para el perón solito para contectos a pueblos de los partidos de la callesa del desarrollo de la pedecencer aquestros ladora del estado!"
(14)   #Diwee AsayMañ@user Pero me el ino que ataca? 🤔enamos el su único va historia@user @user @user A mí va wn en menos más en lo q es culiao ha sido q lo surgencia a esta la wea y hay simbólicas y cambia de comer el agua feminista nada que vine la larg
(15)  re @userveeel con la loco nefastos...   Una mujer qlioo se hueve@user También no les chileno pasa... 🤣 Una compensabrón que no tenemos nada falta ni #Chile #Bretis #nadiesque https://t.co/THlLNp6lYInglo basta apruebora de gran colomando a la CON EL 
(16)                              @user   SOS,KAST A UN PARTE@user Je veneces un estanto hacilo grande k no de aquí no hay vavisimos a 6 de conscribes se jode que fue me muestrar más gana usar el cabezo@user @user Estadió más mi corrupto, sino que no es i
(17)  😒2. https://t.co/Le5rwySLFEstal del ministro hombre o mas https://t.co/R73C6EIzmLLas feministas semanas decisiones en santos del reglesento. https://t.co/0Bzt4Pk8WT@user No debería existir el mundo en que solo un delincuente para escondiGnocuador ha
(18)  A ¿?  ¿sé Gracia? El charceste año por cagar, que usted la derroga?"@user @user @user @user https://t.co/lVI7wjhkgR9No veo se necesita el trampleo que los golpe es de que no son demostrar combalando en toda esta gente no eres negro ni respetada. No 
(19)     El bis en el nuevo inflacereta k sientoma a esa wea ctm https://t.co/8t6zJfi8JHajajajajaja si se sarto es bueno hace otros muertos países con el DT eaaaaaaaaas que no tienes a la izquierda y metieron los hombres.@user Hijo de puta con los artific
(20)      #LaFamilio hajajajajaja no lo hizo más renuncio habla bla hacer peruanos lo mismo beha del reflejo sexual no 🤣 https://t.co/ANqSpuXPZ@user Pero es el respeto que se hay un pidistito y nuevo gran compatriotas. No sé que alguna muñeca son totalmen
```

## acknowledgements

El código fue adaptado del proyecto [nanoGPT](https://github.com/karpathy/nanoGPT).

