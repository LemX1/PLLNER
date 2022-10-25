   项目结构
   │

     ——preprocess

           ——easydata_2_conll.py　　#将数据从easydata格式转换成CONLL形式

           ——easydata_2_partial.py　　#将数据从easydata格式转换成预设的partial label形式

           ——extract_subset.py　　#将partial label的训练集进行切分

           ——generate_vote_data.py　　#生成投票数据

           ——divide_by_similarity.py　　#按标注结果统一程度对数据集进行切分

           ——split.py　　#切分数据，默认按8:1:1切分成train/dev/test

     ——model

           ——bio_ent2id.json　　#bio编码下的实体id映射关系，字典结构

           ——bioes_ent2id.json　　#bioes编码下的实体id映射关系，字典结构

           ——model.py　　#模型结构

     ——utils

           ——processor.py　　#先将原始输入转化为bert的features，再将features转化成pytorch的Dataloader可以处理的dataset

            ——decode.py　　#将预测序列解码成dict（span）格式

           ——CSIDN.py　　#计算/运用转移矩阵所需函数

           ——args.py　　#配置超参数和获取超参数

           ——set_seed.py　　#设置随机种子

           ——trainer.py　　#训练器类，用于训练，保存训练后的模型

           ——metics.py　　#计算模型评分

           ——evaluate.py　　#在所给数据集上评估所给模型

           ——load_model_and_parallel.py　　#从本地加载模型（如果输入是路径的话），然后将模型加载到GPU上

           ——LWLoss.py　　#Leverage Loss相关函数

      ——solely_test.py　　#用于测试本地模型

      ——run.py   #主函数
