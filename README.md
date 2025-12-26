# MNIST-Neural-Network-Numpy - MNIST Digit Classification

## 🎯 Sobre o Projeto

Este projeto foi desenvolvido com o objetivo de **entender profundamente a matemática por trás de Machine Learning e Neural Networks**. Ao invés de usar frameworks de alto nível, implementei toda a rede neural do zero usando apenas NumPy, fazendo todas as contas na mão - forward propagation, backpropagation, cálculo de gradientes, etc.

Foi uma jornada incrível de aprendizado onde pude realmente compreender o que acontece "por baixo do capô" de modelos de deep learning! 🧠📐

## 📋 Descrição

Uma rede neural feedforward de 3 camadas implementada do zero para classificar dígitos manuscritos do dataset MNIST. A implementação inclui:

- **Arquitetura**: 784 → 64 → 32 → 10 neurônios
- **Funções de ativação**: ReLU (camadas ocultas) e Softmax (saída)
- **Técnicas de regularização**: Dropout e Weight Decay (L2)
- **Otimização**: Gradient Descent com mini-batches
- **Acurácia**: ~97-98% no conjunto de teste

## 🚀 Funcionalidades

- ✅ Implementação manual de forward e backward propagation
- ✅ Dropout para regularização
- ✅ Weight Decay (L2 regularization)
- ✅ Mini-batch gradient descent
- ✅ Inicialização He para pesos
- ✅ Visualização de predições (corretas e erradas)
- ✅ Gráficos de loss durante treinamento

## 📊 Arquitetura da Rede

```
Input Layer:    784 neurônios (28x28 pixels)
                  ↓
Hidden Layer 1:  64 neurônios (ReLU + Dropout)
                  ↓
Hidden Layer 2:  32 neurônios (ReLU + Dropout)
                  ↓
Output Layer:    10 neurônios (Softmax)
```

## 🛠️ Tecnologias Utilizadas

- Python 3.13.2
- NumPy (computação numérica)
- Matplotlib (visualização)
- TensorFlow/Keras (apenas para carregar o dataset MNIST)

## 📦 Instalação

```bash
# Clone o repositório
git clone https://github.com/EduardoTBuss/MNIST-Neural-Network-Numpy
cd MNIST-Neural-Network-Numpy

# Instale as dependências
pip install -r requirements.txt
```

## 🎮 Como Usar

```bash
python main.py
```

O script irá:
1. Carregar e preprocessar o dataset MNIST
2. Treinar a rede neural por 35 épocas
3. Salvar gráficos de loss a cada época
4. Exibir a acurácia final no conjunto de teste
5. Mostrar exemplos de predições (corretas e incorretas)

## ⚙️ Hiperparâmetros

```python
LR = 0.006              # Taxa de aprendizado
batch_size = 16         # Tamanho do mini-batch
epochs = 35             # Número de épocas
weight_decay = 0.001    # Regularização L2
dropout1 = 0.1          # Dropout primeira camada
dropout2 = 0.1          # Dropout segunda camada
```

## 📈 Resultados Esperados

- **Acurácia de Treinamento**: ~98-99%
- **Acurácia de Teste**: ~97-98%
- **Loss Final**: ~0.07-0.10

## 🔍 Componentes Principais

### Forward Propagation
Calcula as ativações de cada camada:
```
Z1 = W1·X + B1
A1 = ReLU(Z1)
Z2 = W2·A1 + B2
A2 = ReLU(Z2)
Z3 = W3·A2 + B3
A3 = Softmax(Z3)
```

### Backward Propagation
Calcula os gradientes usando a regra da cadeia:
```
dL/dW = (1/m) · dZ · A^T
dL/dB = (1/m) · Σ(dZ)
```

### Regularização
- **Dropout**: Desativa aleatoriamente neurônios durante o treino
- **Weight Decay**: Adiciona penalidade L2 aos pesos

## 📁 Estrutura de Arquivos

```
.
├── main.py                # Script principal
├── requirements.txt       # Dependências do projeto
├── grafico_loss.png       # Gráfico de loss (atualizado a cada época)
├── LICENSE                # Licença MIT
└── README.md              # Este arquivo
```

## 🎨 Visualizações

O código gera:
- **Gráfico de Loss**: Mostra a evolução do loss durante o treinamento
- **Grid 3x3 de Predições**: Mostra imagens com labels verdadeiros e preditos
- **Grid 3x3 de Erros**: Mostra apenas exemplos onde a rede errou

## 🧮 O que Aprendi

Implementar tudo do zero me permitiu entender:
- Como funciona a backpropagation matematicamente
- A importância da inicialização de pesos
- Como o dropout previne overfitting
- O papel da regularização L2
- Como otimizadores atualizam os parâmetros
- A diferença entre gradientes no treino e na inferência

## 🤝 Contribuições

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues ou pull requests.

## 📝 Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para mais detalhes.

---

**⭐ Se este projeto te ajudou a entender melhor Neural Networks, considere dar uma estrela!**