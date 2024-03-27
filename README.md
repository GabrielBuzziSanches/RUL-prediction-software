# Software para predição da vida útil remanescente de baterias Li-ion

Este software inclui:  
- Dados brutos de baterias extraídos em laborário para treinamento do modelo 
- Modelos treinados para inferência  
- Código geração dos recursos de dados de entrada do modelo
- Código para treinamento do modelo
- Código para teste do modelo
- Pipeline para inferência em novos dados

Os dados processados já estão incluídos no repositório na pasta "./data/features/", portanto não há necessidade de baixar os dados raw e gerar as features novamente para treinar e avaliar o modelo.

Mas caso haja interesse em reproduzir as features os dados brutos utilizados estão disponibilizados no link:

https://drive.google.com/drive/folders/1e7yYJK_JIK2QL8fGDCnt26ciSaEjpZUI?usp=sharing

Após o download os arquivos .pkl de cada partição (train, test, val) devem ser colocados nas respectivas pasta em "./data/raw/".  

Os modelos treinados estão no arquivo .zip "models.zip". Então não é necessário rodar o arquivo de treinamento caso desejar apenas testar o modelo, apenas descompactar essa pasta na raiz do projeto. Caso queira treinar novamente o modelo novas versões dos arquvios serão salvas na pasta "./models".

### Aplicação  
Para aplicar o software em campo incluir os dados coletados em um unico ciclo de teste executado em um bateria desconhecida na pasta "./data/inference/".
Recomenda-se que os dados estejam em um formato .csv com as colunas: V, I, Qc, Qd, T, t, IR, chargetime; mas caso a estrutura não seja essa será necessário alterar o código no arquivo pipeline_inferencia.py no local sinalizado para carregar esses dados nas respectivas variáveis.  
Existe um arquivo dentro da pasta "./data/inference/" com um exemplo do formato ideal de dados para a aplicação do modelo.

Uma vez que os dados forem carregados basta executar o arquivo pipeline_inferencia.py para obter a estimativa da vida útil remanescente da bateria (RUL).