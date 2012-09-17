import numpy as np
import cPickle

from rlscore import data_sources

    
def write_dense(fname, data):
    """Writes an array of floats as a dense text file.
    
    Parameters
    ----------
    fname : string
        output file name
        
    data : array-like
        data to be written
    """
    np.savetxt(fname, data)
    
def write_sparse(fname, data):
    """Writes an array of floats in the sparse text format.
    
    Warning: not optimized for sparse matrices (will iterate
    also over zero values)
    
    Parameters
    ----------
    fname : string
        output file name
        
    data : {array-like, sparse matrix}
        data to be written
    """
    f = open(fname, 'w')
    rlen, clen = data.shape
    for i in range(rlen):
        ss = ''
        for j in range(clen):
            if data[i, j] != 0:
                ss += ' ' + str(j) + ':' + str(data[i, j])
        ss = ss.strip()
        f.write(ss + '\n')
        f.flush()
    f.close()
    
def write_ints(fname, data):
    """Writes an array of integers as a dense text file.
    
    Parameters
    ----------
    fname : string
        output file name
        
    data : array-like
        data to be written
    """
    np.savetxt(fname, data, fmt='%i')
    
def write_numpy(fname, data):
    """Writes an array of floats as a dense text file.
    
    Parameters
    ----------
    fname : string
        output file name
        
    data : {array-like, sparse matrix}
        data to be written
    """
    if isinstance(data, sparse.spmatrix):
        data = data.todense()
    np.save(fname, data)
    
def write_pickle(fname, data):
    """Pickles an object to the disk.
    
    Parameters
    ----------
    fname : string
        output file name
        
    data : object
        data to be written
    """
    f = open(fname,'wb')
    cPickle.dump(data, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()
    
DEFAULT_WRITERS = {
                   'matrix': write_dense,
                   'model': write_pickle,
                   data_sources.INT_LIST_TYPE: write_ints,
                   data_sources.FLOAT_LIST_TYPE: write_dense
                   }