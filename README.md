# Objectif

L'objectif de mon projet est de déterminer si une feuille malade permet de prédire si la plante risque de mourir ou si elle produira des fruits de mauvaise qualité. D'après mes recherches, une plante présentant des taches, des trous ou des couleurs anormales indique une maladie et un risque de mort imminent. En ce qui concerne les fruits, une plante malade produit souvent des fruits de mauvaise qualité, voire non comestibles. Ainsi, en prédisant l'état d'une plante malade, on peut limiter sa propagation et, idéalement, la guérir. L'objectif de ce dataset est d'analyser les feuilles d'une plante afin de prédire leur état, ce qui pourrait être utile dans un potager, comme dans le cas de mon père qui cultive des tomates. Cela me permettrait de reconnaître si mes feuilles de tomate malades risquent de donner des tomates de mauvaise qualité.

# Dataset

Dans un premier temps, après avoir récupérée notre dataset depuis kaggle sous format zip. Le dataset classe 17 plantes malades ainsi que 10 plantes en bonne santé. Il est constitué de deux dossiers : un dossier 'train' contenant 2516 images et un dossier 'test' également composé de 2516 images.

# Entraînement du modèle

Nous avons utilisé un CNN pour la prédiction d’images, car il permet de traiter les images en tant que données pour la classification. Pour cela, nous avons utilisé TensorFlow et Keras : TensorFlow est une bibliothèque d'apprentissage automatique, tandis que Keras est une interface de haut niveau qui facilite la création de modèles d'apprentissage profond. Nous avons également utilisé Google Colab pour exécuter le notebook, car il est plus efficace que nos machines et nous permet d'entraîner nos modèles plus rapidement.

## Architecture du CNN
```python
model = Sequential([
  layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),

  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),

  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),

  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),

  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])
```

## Précision



## Résultat 



