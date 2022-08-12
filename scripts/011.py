from torch.utils.data import DataLoader, get_worker_info
import pickle
def getVocab(file):
    """
        get Vocabulary from pkl file  从pkl文件获取词汇
    """
    g = open(file,'rb')
    dic = pickle.load(g)
    g.close()
    return dic

#file=open("/NPA/Codes/News-Recommendation/data/dictionaries/vocab_title.pkl","rb")
#data=pickle.load(file)
#embedding = GloVe(dim=300,cache='/NPA/Data/MIND/vector_cache')#将此矩阵编码为向量矩阵
vocab = getVocab(r'E:\NEWS\Codes\News-Recommendation\data\dictionaries\vocab_title.pkl')
#vocab.load_vectors(embedding)#加载词向量矩阵
c=1
a=c==1
print('aaaa')