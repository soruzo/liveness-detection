
# Liveness Detection

Este projeto implementa um sistema de detecção de piscadas para verificar a presença de um usuário real em frente à câmera, como parte de um sistema de liveness detection.

---

### Integrantes:
- Anthony Carvalho
- Bruno Eduardo
- Bruno Neves
- Carla Scherer

---

## Estrutura do Projeto
- `main.py`: Script principal para detecção de piscadas.
- `models/`: Contém modelos pré-treinados do `dlib` para detecção de características faciais.
- `README.md`: Este arquivo com instruções e informações sobre o projeto.

---

## Configurando o Ambiente
#### Requisitos:
- Python 3.x
- Bibliotecas Python: `opencv-python`, `dlib`, `numpy`

### Dependências do Sistema Operacional

#### Windows:
- CMake: Necessário para a compilação do `dlib`. Pode ser instalado via [CMake](https://cmake.org/download/).
- Visual Studio com C++: Necessário para compilar dependências do Python. Disponível em [Visual Studio](https://visualstudio.microsoft.com/pt-br/).

#### Linux/MacOS:
- CMake e uma ferramenta de build (como `make`): Necessários para a compilação do `dlib`. Normalmente disponíveis nos repositórios padrão da maioria das distribuições Linux e podem ser instalados via gerenciador de pacotes.
   ```bash
   # Exemplo para Debian/Ubuntu
   sudo apt-get install cmake make
   ```

#### Instalação das Dependências Python:
```bash
pip install opencv-python dlib numpy
```

---

## Como Executar
1. Execute o script `main.py` para iniciar o sistema de detecção de piscadas.
   ```bash
   python main.py
   ```
2. O programa solicitará que o usuário pisque um número específico de vezes, indicado na tela.
3. Uma contagem regressiva será exibida antes de começar a detecção.
4. O usuário deve piscar o número exato de vezes solicitado para ser validado.

---

## O que esperar?
O script `main.py` usará a webcam para detectar o rosto do usuário e monitorar os movimentos dos olhos. O sistema conta o número de piscadas e valida o usuário se o número de piscadas corresponder ao solicitado. Uma mensagem indicará o sucesso ou falha da validação na tela.

---