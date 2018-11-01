# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 15:23:34 2018

@author: sayan_banerjee03
"""
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.stem.snowball import SnowballStemmer
import numpy as np
import pandas as pd
import re
#import csv
import string
from scipy.sparse import lil_matrix, find
import itertools
from pyjarowinkler import distance
from sklearn.feature_selection import SelectPercentile
import os.path
import hashlib
import pickle
import filelock
from io import StringIO



stemmer = SnowballStemmer("english", ignore_stopwords=True)

class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])

class ItemSelector(BaseEstimator, TransformerMixin):

    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_df):
        return data_df[self.key]
    
    def get_feature_names():
       return []

class TextStats(BaseEstimator, TransformerMixin):
    """Extract features from each document for DictVectorizer"""

    def fit(self, x, y=None):
        return self

    def transform(self, posts):
        return [{'length': len(text),
                 'num_sentences': (text.count('.') + \
                                   text.count('?') + \
                                   text.count('!'))}
                for text in posts]
        
    def get_feature_names(self):
       return ['length','num_sentences']

class TargetSimilarity(BaseEstimator, TransformerMixin):
    """Extract features from each document for DictVectorizer"""
    def __init__(self, target,stop_words = None, ngram_range = (1,3),use_idf = False):
        self.target = target
        self.stop_words = stop_words
        self.ngram_range = ngram_range
        self.use_idf = use_idf
        
    def fit(self, x, y=None):
        return self

    def transform(self, text):
        text_target = np.append(text,self.target)
        count_vect = StemmedCountVectorizer(stop_words = self.stop_words, 
                                     ngram_range = self.ngram_range)
        counts = count_vect.fit_transform(text_target)
        
        # TF-IDF
        tfidf_transformer = TfidfTransformer(use_idf = self.use_idf)
        tfidf = tfidf_transformer.fit_transform(counts)
        #tfidf = TfidfVectorizer().fit_transform(text_target)
        #cosine_similarities = linear_kernel(tfidf[0:1], tfidf).flatten()
        cosine_similarities = (tfidf * tfidf.T).A
        #squareform(pdist(tfidf.toarray(), 'cosine'))
        return cosine_similarities[:-len(self.target),len(text):]
    
class MyModelTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        return self

    def transform(self, X):
        return self.model.predict_proba(X)

class NumberTransformer(BaseEstimator, TransformerMixin):
    """Extract features from each document for DictVectorizer"""

    def fit(self, x, y=None):
        return self

    def transform(self, text_col):
        num_text_col = text_col.replace(to_replace=re.compile('(?:(?<=\s)|(?<=^)|(?<=[^0-9a-zA-Z]))[0-9][0-9.,\-\/]*(?:(?=\s)|(?=$)|(?=[^0-9a-zA-Z]))',flags = re.IGNORECASE),
                    value='NUMBERSPECIALTOKEN',inplace=False,regex=True)
        return num_text_col

# =============================================================================
#         text_col.replace(to_replace=re.compile('(?:(?<=\s)|(?<=^)|(?<=[^0-9a-zA-Z]))[0-9][0-9.,\-\/]*(?:(?=\s)|(?=$)|(?=[^0-9a-zA-Z]))',flags = re.IGNORECASE),
#                     value='NUMBERSPECIALTOKEN',inplace=True,regex=True)
#         return text_col
# =============================================================================
        
    def get_feature_names(self):
       return None

class DateTransformer(BaseEstimator, TransformerMixin):
    """Extract features from each document for DictVectorizer"""

    def fit(self, x, y=None):
        return self

    def transform(self, text_col):
# =============================================================================
#         text_col.replace(to_replace=re.compile('(?:(?<=\s)|(?<=^)|(?<=[^0-9a-zA-Z]))(\d+[/-]\d+[/-]\d+)(?:(?=\s)|(?=$)|(?=[^0-9a-zA-Z]))',flags = re.IGNORECASE),
#                 value='DATESPECIALTOKEN',inplace=True,regex=True)
# =============================================================================
        date_text_col = text_col.replace(to_replace=re.compile('(?:(?<=\s)|(?<=^)|(?<=[^0-9a-zA-Z]))(\d+[/-]\d+[/-]\d+)(?:(?=\s)|(?=$)|(?=[^0-9a-zA-Z]))',flags = re.IGNORECASE),
                 value='DATESPECIALTOKEN',inplace=False,regex=True)
        return date_text_col
        
    def get_feature_names(self):
       return None
   
class SynonymTransformer(BaseEstimator, TransformerMixin):
    """Extract features from each document for DictVectorizer"""

    def __init__(self, synonym_dict):
        self.syn_dict = synonym_dict
        #print(self.syn_dict)
        
    def fit(self, x, y = None):
        #print(self.syn_dict)
        return self

    def transform(self, text_col):
        #print(self.syn_dict)
        date_text_col = text_col.replace(self.syn_dict,regex=True)
        return date_text_col
        
    def get_feature_names(self):
       return None

class PunctTransformer(BaseEstimator, TransformerMixin):
    """Extract features from each document for DictVectorizer"""

    def fit(self, x, y=None):
        return self

    def transform(self, text_col):
        regexp = '['+string.punctuation+']{2,}'
        punct_text_col = text_col.replace(to_replace=re.compile(regexp,
                                                                flags = re.IGNORECASE),
                                             value='',inplace=False,regex=True)
        
        punct_text_col = punct_text_col.replace(to_replace=re.compile('\n+|\r+|\t+',
                                                                flags = re.IGNORECASE),
                                             value=' ',inplace=False,regex=True)
        return punct_text_col
        
    def get_feature_names(self):
       return None

    
class FeaturizeDomainKeyWords(BaseEstimator, TransformerMixin):
    """Extract features from each document for DictVectorizer"""

    def __init__(self, domain_keyword_list = []):
        self.keywords_list = domain_keyword_list
        #print(self.keywords_list)
    
    def fit(self, x, y = None):
        #print(self.keywords_list)
        return self

    def transform(self, text_col):
        #print(self.keywords_list)
        keyword_textcol_list = []
        if self.keywords_list != None:
            for text in text_col:
                keyword_textcol_dict = {}
                for keywords in self.keywords_list:
                    if(len(keywords) > 1):
                        keyword_reg = "|".join(keywords)
                    else:
                        keyword_reg = keywords[0]
                    keyword = 'has_keyword_' + keywords[0]
                    keyword_textcol_dict[keyword] = bool(re.search(keyword_reg,text,re.IGNORECASE))
                keyword_textcol_list.append(keyword_textcol_dict)
        return keyword_textcol_list
        
    def get_feature_names(self):
        keyword_col_list = ['has_keyword_' + keywords[0] for keywords in self.keywords_list]
        return keyword_col_list
    
def ClassDiscriminatingMeasure(X,y):
    CDM_tk =np.zeros(shape=(X.shape[1],))
#full_term_sum = tr_dsc_vect.tocsr().sum(0)
    for category in np.unique(y):
        #print(category)
        pos_loc = np.where(y == category)[0]
        cat_num_doc = len(pos_loc)
        #print(cat_num_doc)
        neg_loc = np.where(y != category)[0]
        neg_cat_num_doc = len(neg_loc)
        #print(neg_cat_num_doc)
        cat_term = X.tocsr()[pos_loc,:]
        #(nonzero_rows,nonzero_cols,_)=sparse.find(cat_term)
        tk_ci = np.diff(cat_term.tocsc().indptr)
        P_tk_ci = tk_ci / cat_num_doc
        #cat_term_sum = cat_term.sum(0)
        cat_term_neg = X.tocsr()[neg_loc,:]
        #cat_term_neg_sum = cat_term_neg.sum(0)
        #(nonzero_rows,nonzero_cols)=cat_term_neg.nonzero()
        tk_neg_ci = np.diff(cat_term_neg.tocsc().indptr)
        P_tk_neg_ci = (1 + tk_neg_ci)/ neg_cat_num_doc
        CDM_tk_ci = np.log1p(P_tk_ci/P_tk_neg_ci)
        CDM_tk = CDM_tk + CDM_tk_ci
    #print(CDM_tk.shape)
    return CDM_tk

def get_context_d_tk_w(d, tk, w = 3,token_regex = r"(?u)\b\w\w+\b"):
    #sentence = sentence.split()
    #d = re.split('[\s\-\:]+',d)
    r_splt = re.compile(token_regex)
    d = r_splt.findall(d)
    len_d = len(d)
    tk = tk.split()
    num_words = len(tk)
    r_st = re.compile(r"\b%s\b" % tk[0], re.IGNORECASE|re.MULTILINE)
    r_cmp = re.compile(r"\b%s\b" % ' '.join(tk), re.IGNORECASE|re.MULTILINE)
    for i,word in enumerate(d):
        if bool(r_st.match(word)) and \
        bool(r_cmp.match(' '.join(d[i:i+num_words]))):
            #print(i)
            #print(word)
            begin_pad = []
            end_pad = []
            if (i-w < 0):
                for b in reversed(range(0,w-i)):
                    begin_pad.append('__START_'+ str(b) +'__')
            #print(begin_pad)
            if (i+num_words+w > len_d):
                for e in range(0,i+num_words+w - len_d):
                    end_pad.append('__END_'+ str(e) +'__')
            #print(end_pad)
            start = max(0, i-w)
            #print(d[start:i+num_words+w])
            begin_pad.extend(d[start:i+num_words+w])
            #print(begin_pad)
            begin_pad.extend(end_pad)
            yield ' '.join(begin_pad)

def pairs(*lists):
    for t in itertools.combinations(lists, 2):
        for pair in itertools.product(*t):
            yield pair
            
def get_sim_context_d_tk_w(docs, tk, m_w = 3):
    sim_context_all_w = []
    for w in reversed(range(0,m_w + 1)):
        doc_contexts = []
        doc_contexts_itr = docs.apply(get_context_d_tk_w,args = (tk,w))
        doc_context_num = []
        for context in doc_contexts_itr:
            list_context = list(context)
            doc_context_num.append(len(list_context)) 
            doc_contexts.append(list_context)
        sim_context_w = []
        for x in pairs(*doc_contexts):
            sim_context_w.append(distance.get_jaro_distance(x[0],x[1]))
        sim_context_all_w.append(sim_context_w)
    sim_context_all_w = np.asarray(sim_context_all_w)
    sim_context_all_w = sim_context_all_w.sum(0)/ (m_w + 1)
    #range_list = []
    sim_context_d_tk_w = []
    for i in range(len(doc_context_num)):
        cr = 0
        range_list = []
        for pair in itertools.combinations(list(range(len(doc_context_num))),2): 
            #print(pair)
            last_pos = cr + (doc_context_num[pair[0]] * doc_context_num[pair[1]])
            all_pos = list(range(cr,last_pos))
            #print(all_pos)
            cr = last_pos      
            if i in pair:
                range_list.extend(all_pos)
        #range_list.append(tmp_range_list)
        sim_context_d_tk_w.append(sum(sim_context_all_w[range_list]))
    return(sim_context_d_tk_w)
       
def ClassDiscriminatingMeasureCS(X,y):
    CDM_tk =np.zeros(shape=(X.shape[1],))
#full_term_sum = tr_dsc_vect.tocsr().sum(0)
    for category in np.unique(y):
        #print(category)
        pos_loc = np.where(y == category)[0]
        cat_num_doc = len(pos_loc)
        #print(cat_num_doc)
        neg_loc = np.where(y != category)[0]
        neg_cat_num_doc = len(neg_loc)
        #print(neg_cat_num_doc)
        cat_term = X.tocsr()[pos_loc,:]
        #(nonzero_rows,nonzero_cols,_)=sparse.find(cat_term)
        tk_ci = cat_term.sum(0)
        P_tk_ci = tk_ci / cat_num_doc
        #cat_term_sum = cat_term.sum(0)
        cat_term_neg = X.tocsr()[neg_loc,:]
        #cat_term_neg_sum = cat_term_neg.sum(0)
        #(nonzero_rows,nonzero_cols)=cat_term_neg.nonzero()
        tk_neg_ci = cat_term_neg.sum(0)
        P_tk_neg_ci = (1 + tk_neg_ci)/ neg_cat_num_doc
        CDM_tk_ci = np.log1p(P_tk_ci/P_tk_neg_ci)
        CDM_tk = CDM_tk + CDM_tk_ci
    #print((CDM_tk.A1).shape)
    return  CDM_tk.A1

def get_sim_context_tk_w(terms,
                       count_vect_obj,
                       raw_document,
                       max_window = 3,
                       token_regex = r"(?u)\b\w\w+\b",
                       stop_words = None,
                       cache_dir = None):
                                           #'(?u)\\b\\w\\w+\\b'):
    cache_dict = {}
    is_cache = False
    cache_update = False
    if (cache_dir != None and os.path.isdir(cache_dir)):
        is_cache = True
        file_sign_str = (raw_document.str.cat(sep = ' ') + str(max_window)).encode(encoding = 'utf-8')
        hash_object = hashlib.md5(file_sign_str)
        cache_file_path = cache_dir + '/' + hash_object.hexdigest() + '.pkl'
        if os.path.isfile(cache_file_path):
            with open (cache_file_path, 'rb') as fp:
                cache_dict = pickle.load(fp)
    term_list = count_vect_obj.get_feature_names()
    raw_document.index = range(len(raw_document))
    #r_splt = re.compile("%s" % token_regex)
    data_lower = raw_document.str.lower().str.findall(token_regex)
    if stop_words != None:
        data_lower_stop = data_lower.apply(lambda x: ' '.join([item for item in x if item not in stop_words]))
    else:
        data_lower_stop = data_lower
    nz_rows, nz_cols, nz_val = find(terms) #.nonzero()
    num_terms = terms.shape[1]
    ret_mat = lil_matrix(terms.shape)
    for term_idx in range(0,(num_terms-1)):
        #print(term_idx)
        term_doc_indx = nz_rows[np.where(nz_cols == term_idx)[0]]
        #nz_val[np.where(nz_cols == term_idx)[0]]
        if (len(term_doc_indx) == 1):
            ret_mat[term_doc_indx,term_idx] = 0 # should this be 1 instead as unique term
        else:
           tk =  term_list[term_idx]
           docs = data_lower_stop[term_doc_indx]
           if len(cache_dict) > 0 and tk in cache_dict:
               sim_context_d_tk_w = cache_dict[tk]
           else:    
               sim_context_d_tk_w = get_sim_context_d_tk_w(docs,tk,max_window)
               if is_cache:
                   cache_dict[tk] = sim_context_d_tk_w
                   cache_update = True
           for i,row_idx in enumerate(term_doc_indx):
               ret_mat[row_idx,term_idx] = sim_context_d_tk_w[i]
    
    if is_cache and cache_update:
        lock = filelock.FileLock("{}.lock".format(cache_file_path))
        try:
            with lock.acquire(timeout = 10):
                with open(cache_file_path, 'wb') as fp:
                    pickle.dump(cache_dict, fp)
        except lock.Timeout:
            print('update_cache timeout' + cache_file_path)
    #CDM_tk = ClassDiscriminatingMeasure(ret_mat,y,'sum')
    return ret_mat


class ContextSimilarityBasedFeatureSelection(CountVectorizer):
    def __init__(self,max_window = 3,
                 input='content', encoding='utf-8',
                 decode_error='strict', strip_accents=None,
                 lowercase=True, preprocessor=None, tokenizer=None,
                 stop_words=None, token_pattern=r"(?u)\b\w\w+\b",
                 ngram_range=(1, 1), analyzer='word',
                 max_df=1.0, min_df=1, max_features=None,
                 vocabulary=None, binary=False, dtype=np.int64,
                 percentile = 10,cache_dir = None):
        super(ContextSimilarityBasedFeatureSelection, self).__init__(input,
             encoding, decode_error, strip_accents, lowercase , preprocessor,
             tokenizer, stop_words, token_pattern, ngram_range ,
             analyzer, max_df, min_df, max_features, vocabulary, binary, 
             dtype)
        self.max_window = max_window
        self.token_pattern = token_pattern
        self.stop_words = stop_words
        #self.percentile = percentile
        self._red_dim = SelectPercentile(score_func=ClassDiscriminatingMeasureCS,
                                         percentile = percentile)
        self.cache_dir = cache_dir
        
    @property
    def percentile(self):
        return self._red_dim.percentile

    @percentile.setter
    def percentile(self, value):
        self._red_dim.percentile = value
    
    @property
    def score_func(self):
        return self._red_dim.score_func

    @score_func.setter
    def score_func(self, value):
        self._red_dim.score_func = value
# =============================================================================
#     def fit(self, raw_documents, y=None):
#         return self
# =============================================================================

    def fit_transform(self, raw_documents, y=None):
        dtm = super(ContextSimilarityBasedFeatureSelection, self).fit_transform(raw_documents)
        sim_context_tk_w = get_sim_context_tk_w(terms = dtm,
                       count_vect_obj = super(ContextSimilarityBasedFeatureSelection, self),
                       raw_document = raw_documents,
                       max_window = self.max_window,
                       token_regex = self.token_pattern,
                       stop_words = self.stop_words,
                       cache_dir = self.cache_dir)
        self._red_dim.fit_transform(sim_context_tk_w,y)
        self.selected_cols = self._red_dim.get_support(indices=True)
        return dtm[:,self.selected_cols]

    def transform(self, raw_documents, copy=True):
        #check_is_fitted(self, '_tfidf', 'The tfidf vector is not fitted')
        dtm = super(ContextSimilarityBasedFeatureSelection, self).transform(raw_documents)
        return dtm[:,self.selected_cols]
    
    def get_feature_names(self):
        all_features = super(ContextSimilarityBasedFeatureSelection, self).get_feature_names()
        return np.asarray(all_features)[self.selected_cols]

def classifaction_report_df(report):
    report = re.sub(r" +", " ", report).replace("avg / total", "avg/total").replace("\n ", "\n")
    report_df = pd.read_csv(StringIO("Classes" + report), sep=' ', index_col=0)        
    return(report_df)
# =============================================================================
#     report_data = []
#     lines = report.split('\n')
#     for line in lines[2:-3]:
#         row = {}
#         row_data = line.split('      ')
#         row['class'] = row_data[0]
#         row['precision'] = float(row_data[1])
#         row['recall'] = float(row_data[2])
#         row['f1_score'] = float(row_data[3])
#         row['support'] = float(row_data[4])
#         report_data.append(row)
#     dataframe = pd.DataFrame.from_dict(report_data)
# #    dataframe.to_csv('classification_report.csv', index = False)
#     return dataframe
# =============================================================================

def get_used_features(mod,explicit_feature_selection = True):
    if explicit_feature_selection:
        mod_support = mod.named_steps['reduce_dim'].get_support(indices=True)
    features = []
    for trnf_list in mod.named_steps['union'].transformer_list:
        features.extend(trnf_list[1].named_steps['vect'].get_feature_names())
    if explicit_feature_selection:
        return(np.asarray(features)[mod_support])
    else:
        return(np.asarray(features))

def get_grid_values(gs_obj):
    means = gs_obj.cv_results_['mean_test_score']
    stds = gs_obj.cv_results_['std_test_score']
    col_name = ['means', 'stds']
    col_name.extend(list(gs_obj.cv_results_['params'][0].keys()))
    perf_df = pd.DataFrame(columns=col_name )    
    i = 0   
    for mean, std, params in zip(means, stds, gs_obj.cv_results_['params']):
# =============================================================================
#         print("%0.3f (+/-%0.03f) for %r"
#                   % (mean, std * 2, params))
# =============================================================================
        row_list = [mean,std * 2]   
        row_list.extend(params.values())
        perf_df.loc[i] = row_list
        i += 1
    return perf_df

def confusion_matrix_df(y_actu,y_pred):
    y_actu = pd.Series(y_actu, name='Actual')
    y_pred = pd.Series(y_pred, name='Predicted')
    #return np.array2string(confusion_matrix, separator=', ')
    df_confusion = pd.crosstab(y_actu, y_pred, 
                               rownames=['Actual'], colnames=['Predicted'], 
                               margins=True)
    return df_confusion