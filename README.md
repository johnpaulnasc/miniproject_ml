## Mini-Projeto Reconhecimento de Dígitos

### 1. Introdução

  O reconhecimento de dígitos escritos a mão é um problema clássico de classificação na área de visão computacional. O problema consiste em receber uma imagem de um número escrito  mão, codificada em tons de cinza, e classificar o dígito decimal (0-9) ali contido. Para estudantes e pesquisadores das técnicas de aprendizado de máquina, o dataset MNIST, cujos exemplos de instâncias estão ilustrados na Figura 1, é utilizado para comparação de técnicas, competições e construções de novas soluções.

<p align="center">
  <img src="https://github.com/user-attachments/assets/c23d82c2-f6a6-49b3-b7e4-182f33f560af" alt="Figura 1. Dataset MNIST com imagens dos dígitos escritos a mão." />
</p>

### 2. Dataset MNIST Adaptado

  Os arquivos $train.csv$ e $test.csv$ contêm imagens do dataset MNIST, em escala de cinza, dos dígitos 0, 1, 4 e 5 escritos a mão. Cada imagem é composta por 28 linhas e 28 colunas em um total de 784 pixels. Cada pixel possui um valor associado único, que indica seu tom de cinza. Quanto mais alto é esse valor, mais escuro é o pixel. Os valores de cada pixel estão no intervalo fechado [0, 255].

  Os dados de entrada, $(train.csv)$, possuem 785 colunas. A primeira coluna, chamada $“label”$ é o dígito que foi desenhado pelo usuário. O resto das colunas contém os valores dos pixels da imagem associada. 

  Cada coluna de pixel, nos dados de treino, é nomeada como $“pixel𝑥”$, onde 𝑥 é um inteiro no intervalo [0, 783]. Para localizar este pixel na imagem, suponha que decompomos 𝑥 como 𝑥 = 𝑖 ∗ 28 + 𝑗, onde 𝑖 e 𝑗 são inteiros no intervalor [0, 27]. Então o “pixel𝑥” está localizado na linha 𝑖 e coluna 𝑗 de uma matriz 28𝑥28 (indexada por zero). Por exemplo, $“pixel31”$ indica o valor do pixel que está na quarta coluna, da esquerda pra direita, e na segunda linha. Os dados de teste, $(test.csv)$, possuem o mesmo formato dos dados de treinamento.

