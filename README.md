# Predicción mediana del precio de NFT's usando LSTM

El surgimiento de la tecnología blockchain, los contratos inteligentes y en general el mundo de las criptomonedas ha generado un nuevo mercado de distintos objetos como imagenes o terrenos virtuales. Estos objetos llamados  tokens suelen estar caracterizados por ser únicos, es decir, que solo tienen un propietario y por pertenecer a un contra tointeligente.

En este proyecto se propone generar un modelo que permita calcular la mediana de los tokens de un contrato con el fin de obtener informacion sobre la variabilidad de los precios de estos tokens. El objetivo consiste en entrenar una red neuronal con información de todas las transacciones del contrato hasta el día anterior, para predecir la mediana de las transacciones del día actual. Esto se hace, ya que si se sabe que mediana tendrán las transacciones del día de hoy, los potenciales compradores tendrán cierta información del precio general de los tokens.

El proyecto cuenta con los siguientes recursos:
- link del [documento](https://github.com/U-dot/NFT-price-predictor/blob/main/paper.pdf)
- link del [video](https://www.youtube.com/watch?v=tKAJgUnLfyU&feature=youtu.be)
- link [notebook interactivo](http://34.125.227.157:8080/notebooks/work/Predicci%C3%B3n%20del%20precio%20de%20NFT's%20.ipynb/?token=ecd742482430bd26924140ed338f1e1164a01e2414968da54bdf7fb76c63a74e)

## Getting Started

Con el anterior link se puede correr en un solo click el cuaderno desde el servidor de jupyter. 
Alternativamente, las siguientes instrucciones permiten correr localmente el cuaderno interactivo.

### Prerequisites

Requerimientos para el software y otras herramientas
- [install jupyter](https://jupyter.org/install)
- install requirements

### Installing


Instalas los requerimientos

    cd /notebook
    pip install -r requirements.txt

Correr
    
    jupyter notebook 


    
## Autores

  - **David Núñez** 
  - **Fabian Andrés Ruiz**
  - **María Sol Botello**
  - **Sofia Salinas Rico**



