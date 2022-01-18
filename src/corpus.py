import os
import re
import json
import scipy.sparse as sp
import numpy as np
from random import sample 
#Author: Antoine Gourru, from Edouard Delasalles cf https://github.com/edouardelasalles/dar   

class Corpus:
    def __init__(self, docs, id2aut,T,data_dir,full_data):
        self.docs = docs
        self.id2aut = id2aut
        self.aut2id = dict(zip([*id2aut.values()],[*id2aut]))
        self.doc2id = dict(zip([*docs],list(range(len([*docs])))))
        self.nt = T
        self.nd = len(docs)
        self.na = len(id2aut)
        self.data_dir = data_dir
        self.full_data = full_data

    def get_text(self,heidi=None):
        if heidi is None:
            trainset = [d['texts'][0] for d in [*self.docs.values()]]
        else:
            trainset = [d['texts'][0] for d in list(filter(lambda x: x['id'] in heidi, self.docs.values()))]
        return trainset
    def get_embedding(self):
        trainset = [d['embedding'] for d in [*self.docs.values()]]
        return trainset
    def get_mask(self):
        trainset = [d['mask'] for d in [*self.docs.values()]]
        return trainset

    def add_embedding(self,embedding_dic):
        temp = np.vstack([*embedding_dic.values()])
        self.mean = np.mean(temp,axis = 0)
        self.var = np.var(temp,axis = 0)
        self.d = self.mean.shape[0]
        for j,emb in embedding_dic.items():
            self.docs[j]['embedding'] = emb
            
    def add_mask(self,embedding_dic):
        temp = np.vstack([*embedding_dic.values()])
        self.mean = np.mean(temp,axis = 0)
        self.var = np.var(temp,axis = 0)
        self.d = self.mean.shape[0]
        for j,emb in embedding_dic.items():
            self.docs[j]['mask'] = emb       
                       
    @classmethod
    def read_data(cls,data_dir):
        fpath = os.path.join(data_dir, 'corpus.json')
        print(f'Loading corpus at {fpath}...')
        docs = {}
        times = set()
        authors = set()
        with open(fpath, 'r',encoding="utf-8") as f:
            for l in f.read().splitlines():
                ex = json.loads(l)
                docs[ex['id']] = ex
                times.add(ex['timestep'])
                for i in ex['authors']:
                    authors.add(i) 
                                
        T= len(times)
        id2aut = dict(zip(list(range(len(authors))),list(authors)))
        aut2id = {v: k for k, v in id2aut.items()}
        full_data = {} 
        for i in [*id2aut]:
            full_data[i] = {}
        for k,dat in docs.items():
            t = dat['timestep']
            for aut in dat['authors']:
                i = aut2id[aut]
                j = full_data[i].get(t)
                if j is not None:
                    full_data[i][t].append(dat['id'])
                else:
                    full_data[i][t] = []
                    full_data[i][t].append(dat['id'])        

        return cls(docs,id2aut,T,data_dir,full_data)


    def get_data_predictionVADE(self):
        aut_tmax = []
        for i,dat_aut in self.full_data.items():
            aut_tmax.append(max([*dat_aut]))
            
        aut_doc_train = self.get_aut_doc_matrix().copy()
        aut_doc_test = sp.dok_matrix((self.na,self.nd))
        autpool = [*self.id2aut]

        data_pairs = []
        labels = []
        colaborations = []

        doc_tp = []
        doc_t = []
        for key,val in self.docs.items():
            
            j = self.doc2id.get(key)
            t_aut = [self.aut2id.get(i) for i in val.get('authors')]
            t = val.get('timestep')
            
            autpool_temp = autpool.copy()
            for indi in t_aut:
                autpool_temp.remove(indi)
                
            for indi in t_aut:
                if t == aut_tmax[indi]:
                    aut_doc_train[indi,j] = 0
                    aut_doc_test[indi,j] = 1
                    doc_tp.append(j)
                    if len(t_aut) > 1:
                        t_aut_temp = t_aut.copy()
                        t_aut_temp.remove(indi)
                        for col in t_aut:
                            colaborations.append((indi,col))
                        
                else:
                    doc_t.append(j)
                    dat = np.hstack((np.array(indi),np.array(j)))
                    data_pairs.append(dat)
                    labels.append(1)
                    #neg        
                    for neg in range(10):
                        dat = np.hstack((np.array(sample(autpool_temp,1)),np.array(j)))
                        data_pairs.append(dat)
                        labels.append(0)

        aut_doc_test = aut_doc_test.tocsr()
        aut_doc_train.eliminate_zeros()
        print("%d documents in the testing set" % (len(doc_tp)))

        return data_pairs,labels,aut_doc_train,aut_doc_test,colaborations,list(set(doc_tp)),list(set(doc_t))
                
    def get_data_prediction(self):
        out_train = {}
        out_test = {}
        out_test_t = {}
        for i in range(self.na):
            out_train[i] = {}
        compt_pos = 0
        compt_inter = 0
        t_z = []
        id_test = []
        for i,dat_aut in self.full_data.items():
            tt = [*dat_aut].copy()
            tm = max(tt)
            tt.remove(tm)
            tmm1 = max(tt)
            ti = [*dat_aut][0]
            so = 0
            t_z.append(ti)
            for t,dat_t in dat_aut.items():
                if t != tm:
                    #ecart = t - ti
                    out_train[i][so] = np.vstack([self.docs[doc]["embedding"] for doc in dat_t])

                    if t == tmm1:
                        out_test_t[i] = np.vstack([self.docs[doc]["embedding"] for doc in dat_t])
                        
                    #if ecart > 1:
                    #    compt_inter+=1
                        
                    so += 1
                    ti = t
                else:
                    out_test[i] = np.vstack([self.docs[doc]["embedding"] for doc in dat_t])
                    [id_test.append(self.doc2id[j]) for j in dat_t]
                                  
        #print("interpolated %d empty timestep" % (compt_inter))

        id_test = list(set( id_test ))
        aut_doc_test = self.get_aut_doc_matrix().copy()[:,id_test]
        
        return out_train,out_test,out_test_t,t_z,aut_doc_test,id_test
    
    def get_data_predictionARNN(self,idtr = False):
        out = []
        out_test = []
        max_int = 0
        max_outt = 0
        aut_doc_test = self.get_aut_doc_matrix().copy()
        id_test = []
        id_train = []

        for i,dat_aut in self.full_data.items():
            for t,dat_t in dat_aut.items():
                if len(dat_t)> max_int:
                    max_int = len(dat_t)
            
        for i,dat_aut in self.full_data.items():
            out_i = []
            tm = max([*dat_aut])
            for t,dat_t in dat_aut.items():
                if t == tm:
                    out_test.append([(self.doc2id[j]+1) for j in dat_t])
                    if idtr:
                        [id_test.append(j) for j in dat_t]
                    else:
                        [id_test.append(self.doc2id[j]) for j in dat_t]
                else:
                    out_i.append([(self.doc2id[j]+1) for j in dat_t])
                    if idtr:
                        [id_train.append(j) for j in dat_t]
                    else:
                        [id_train.append(self.doc2id[j]) for j in dat_t]
                                 
            if len(out_i)>max_outt:
                max_outt = len(out_i)    
            out.append(out_i)

        if idtr:
            id_test = list(set( id_test ))
            id_train = list(set( id_train ))
            return id_train,id_test
        else:
            A_mask = np.ones((self.na,max_outt,max_int),dtype= np.int64)
            
            for i,l in enumerate(out):
                for j,ll in enumerate(l):
                    A_mask[i,j,len(ll):] = 0
                    [ll.append(0) for k in range((max_int - len(ll)))]    
                A_mask[i,len(l):,:] = 0
                [l.append([0 for k in range(max_int)]) for p in range((max_outt - len(l)))]
                

            A = np.array(out)

            Y_mask = np.ones((self.na,max_int),dtype= np.int64)
            for i,l in enumerate(out_test):
                Y_mask[i,len(l):] = 0
                [l.append(0) for i in range((max_int - len(l)))]
                
                
            Y = np.array(out_test)

            id_test = list(set( id_test ))
            id_train = list(set( id_train ))
            aut_doc_test = aut_doc_test[:,id_test]
            
            return A,A_mask,Y,Y_mask,max_int,max_outt,id_test,aut_doc_test
                
    def get_data_predictionRNN(self):
        out_train = []
        out_test = {}
        out_test_t = {}
        
        t_z = []
        for i,dat_aut in self.full_data.items():
            tt = [*dat_aut].copy()
            tm = max(tt)
            tt.remove(tm)
            tmm1 = max(tt)
            ti = [*dat_aut][0]
            so = 0
            t_z.append(ti)
            for t,dat_t in dat_aut.items():
                if t != ti:                   
                    if t != tm:
                        temp = []
                        for t_2 in [*dat_aut]:
                            if t_2 < t:
                                temp.append(np.mean(np.vstack([self.docs[doc]["embedding"] for doc in dat_aut[t_2]]),axis=0))
                        for doc in dat_t:
                            out_train.append(np.hstack([np.array(i),np.hstack(temp),self.docs[doc]["embedding"]]))
                        if t == tmm1:
                            out_test_t[i] = (np.vstack(temp),np.vstack([self.docs[doc]["embedding"] for doc in dat_t]))
                            
                        so += 1
                    else:
                        temp = []
                        for t_2 in [*dat_aut]:
                            if t_2 < t:
                                temp.append(np.mean(np.vstack([self.docs[doc]["embedding"] for doc in dat_aut[t_2]]),axis=0))
                        out_test[i] = (np.vstack(temp),np.vstack([self.docs[doc]["embedding"] for doc in dat_t]))
                                  
        #print("interpolated %d empty timestep" % (compt_inter))
        return out_train,out_test,out_test_t,t_z
    

    def get_aut_doc_matrix(self):
        aut_doc = sp.dok_matrix((self.na,self.nd))                
        for heid in [*self.docs]:
            j = self.doc2id[heid]
            authors = self.docs[heid]["authors"]
            for aut in authors:
                i = self.aut2id[aut]
                aut_doc[i,j] = 1 
        aut_doc = aut_doc.tocsr()
        return aut_doc

    '''
    
    def split_prediction(self):                     
        idtest = []
        idtrain = []
        for i,dat_aut in self.full_data.items():
            tm = max([*dat_aut])
            for t,dat_t in dat_aut.items():
                if t == tm:
                    idtest += dat_t
                else:
                    idtrain += dat_t

        id_train = list(set(idtrain))
        id_test = list(set(idtest))

        x = abs(self.nd - len(self.id_train) - len(self.id_test))
        print(" %d docments in both test and train test due to coauthorship (%03.2f percent)" % (x,(x/self.nd)*100) )

        with open(os.path.join(self.data_dir, 'train.txt'), 'w+') as f:
            for item in self.id_train:
                f.write("%s\n" % item)   
        with open(os.path.join(self.data_dir, 'test.txt'), 'w+') as f:
            for item in self.id_test:
                f.write("%s\n" % item)

        return id_train,id_test

    
    def split_static(self,ratio):
        self.data = {}
        self.data["train"] = {}
        self.data["test"] = {}
        
        idtrain = []
        idtest = [] 
          
        for i,se in self.full_data.items():
            temp = []
            for t,val in se.items():
                temp += val
            
            if(len(temp)>1):
                npos = int(ratio * len(temp))
                self.data["train"][i] = temp[:npos]
                idtrain+=temp[:npos]
                self.data["test"][i] = temp[npos:]
                idtest+=temp[npos:]
            else:
                self.data["train"][i] = temp
                idtrain+=temp
            
        id_train = list(set(idtrain))
        id_test = list(set(idtest))

        with open(os.path.join(self.data_dir, 'train.txt'), 'w+') as f:
            for item in self.id_train:
                f.write("%s\n" % item)   
        with open(os.path.join(self.data_dir, 'test.txt'), 'w+') as f:
            for item in self.id_test:
                f.write("%s\n" % item)

        return id_train,id_test

    def split_static_val(self,ratio):
        self.data = {}
        self.data["train"] = {}
        self.data["test"] = {}
        self.data["val"] = {}
        
        idtrain = []
        idval = []
        idtest = [] 
          
        for i,se in self.full_data.items():
            temp = []
            for t,val in se.items():
                temp += val
                
            npos = int(ratio[0] * len(temp))
            ntt = len(temp) - npos
            if ntt>0:
                if ntt>1:
                    ntest,nval = divmod(ntt, 2)
            nval = len(temp) - npos + ntest
            if(npos == len(temp)):
                self.data["train"][i] = temp
                idtrain+=temp
                
                npos = int(ratio * len(temp))
                self.data["train"][i] = temp[:npos]
                idtrain+=temp[:npos]
                self.data["test"][i] = temp[npos:]
                idtest+=temp[npos:]
            else:
                self.data["train"][i] = temp
                idtrain+=temp
            
        id_train = list(set(idtrain))
        id_test = list(set(idtest))

        with open(os.path.join(self.data_dir, 'train.txt'), 'w+') as f:
            for item in self.id_train:
                f.write("%s\n" % item)   
        with open(os.path.join(self.data_dir, 'test.txt'), 'w+') as f:
            for item in self.id_test:
                f.write("%s\n" % item)
        return id_train,id_test
   
                
    def read_train(self):

        aut2id = {v: k for k, v in self.id2aut.items()}

        data = {}

        for se in ['train','test']:
            with open(os.path.join(self.data_dir, se+'.txt'), 'r') as f:
                train_ids = set(f.read().splitlines())
            trainset = list(filter(lambda x: x['id'] in train_ids, self.docs.values()))
            train_data = {} 
            for i in [*self.id2aut]:
                train_data[i] = {}
            for dat in trainset:
                t = dat['timestep']
                for aut in dat['authors']:
                    i = aut2id[aut]
                    j = train_data[i].get(t)
                    if j is not None:
                        train_data[i][t].append(dat['id'])
                    else:
                        train_data[i][t] = []
                        train_data[i][t].append(dat['id'])

            data[se] = train_data

        return data
        return id_train,id_test

    def get_temp_data(self,embedding = True):
        maxx = 0
        data = {}
        for i,dat_aut in self.full_data.items():
            temp = []
            for t,dat_t in dat_aut.items():
                if embedding:
                    temp.append(np.vstack([self.docs[doc]["embedding"] for doc in dat_t]))
                else:
                    temp.append(np.vstack([self.docs[doc]["texts"] for doc in dat_t]))
                    
            data[i] = np.vstack(temp)
            if data[i].shape[0] > maxx:
                maxx = data[i].shape[0]
          
        return data,maxx

    def get_data_prediction(self):
        out_train = {}
        out_test = {}
        for i in range(self.na):
            out_train[i] = {}
        compt_pos = 0
        compt_inter = 0
        for i,dat_aut in self.full_data.items():
            tm = max([*dat_aut])
            for t,dat_t in dat_aut.items():
                start = True
                so = 0
                if not start:
                    temp = np.vstack([self.docs[doc]["embedding"] for doc in self.data["train"][i][t]])
                    ecart = t - ti      
                    if ecart > 1:
                        interpol = (np.mean(temp,axis=0) - np.mean(out_train[i][so-1],axis=0))/(ecart)                         
                        for j in range(so,so+ecart-1):
                            out_train[i][so] = np.mean(out_train[i][so-1],axis=0) + np.reshape(interpol,(1,self.d))
                            so+=1
                            compt_inter +=1
                            
                    out_train[i][so] = temp 
                    so += 1
                    ti = t
                    compt_pos +=1
                else:
                    if t == tm:
                        print("bb")
                    else:
                        out_train[i][so] = np.vstack([self.docs[doc]["embedding"] for doc in self.data["train"][i][t]])
                        ti = t
                        so += 1
                        compt_pos +=1
                        start = False
               
        print("interpolated %d va, for a total of %d author/timestep" % (compt_inter,compt_pos))
        return out_train,out_test
    '''
