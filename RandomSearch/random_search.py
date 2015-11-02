import numpy as np
import theano
import theano.tensor as T
import time


def random_search(
    train_probs, val_probs,
    target_train, target_val, numpy_rng, n_iter=400):

    n_modal = train_probs.shape[0]
    n_cls = train_probs.shape[2]
    
    # sample random weights and normalize so the modalities sum to 1
    # for each class
    weight_samples = numpy_rng.rand(n_iter, n_modal, n_cls)
    weight_samples /= weight_samples.sum(1)[:, None, :]
    
    # combine and keep track of best on validation
    best_weights = np.ones((n_modal, n_cls))
    best_iter = -1
    best_acc = 0.0
    for it, weights in enumerate(weight_samples):
        if it%10000 == 0:
            print it
        pred = (val_probs * weights[:, None, :]).sum(0)
        classification = np.argmax(pred, axis=1)
        acc = np.mean(classification == target_val)
        if acc > best_acc:
            best_acc = acc
            best_iter = it
            best_weights = weights
            print 'found new best with accuracy {0}'.format(best_acc)
    
    # TODO: fine-grained search
    return best_weights, best_iter, best_acc

def local_random_search(
    train_probs, val_probs,
    test_probs, test_vidids,
    target_train, target_val, numpy_rng, n_iter,
    mu, std, best_acc, modalities):

    n_modal = train_probs.shape[0]
    n_cls = train_probs.shape[2]
    
    # sample random weights and normalize so the modalities sum to 1
    # for each class
    weight_samples = numpy_rng.normal(loc=mu, scale=std,
                                      size=(n_iter, n_modal, n_cls))
    weight_samples = weight_samples * (weight_samples > 0)
    weight_samples /= weight_samples.sum(1)[:, None, :]
    
    # combine and keep track of best on validation
    best_weights = mu
    best_iter = -1
    for it, weights in enumerate(weight_samples):
        if it%10000 == 0:
            print it
        pred = (val_probs * weights[:, None, :]).sum(0)
        classification = np.argmax(pred, axis=1)
        acc = np.mean(classification == target_val)
        if acc > best_acc:
            # acc for train
            train_pred = (train_probs * weights[:, None, :]).sum(0)
            test_pred = (test_probs * weights[:, None, :]).sum(0)
            train_classification = np.argmax(train_pred, axis=1)
            train_acc = np.mean(train_classification == target_train)
            best_acc = acc
            best_iter = it
            weight_dist = np.linalg.norm(weights-best_weights)
            best_weights = weights
            print 'found new best with accuracy val:{0} train:{1} dist:{2}'.format(
                best_acc, train_acc, weight_dist)
            np.savez('weights/accval{0}_acctrain{1}_{2}_{3}.npz'.format(
                best_acc, train_acc, modalities, time.strftime('%Y-%m-%d-%H-%M')),
                     val_acc=best_acc, train_acc=train_acc, weights=weights,
                     test_pred=test_pred, test_vidids=test_vidids)
    
    return best_weights, best_iter, best_acc

def random_search_gpu(
    modal_names, train_probs, val_probs,
    target_train, target_val, numpy_rng, n_iter=400):

    n_modal = train_probs.shape[0]
    n_cls = train_probs.shape[2]
    
    # sample random weights and normalize so the modalities sum to 1
    # for each class
    
    weight_samples = T.ftensor3('weight_samples')
    probs = T.ftensor3('probs')
    targets = T.ivector('targets')
    preds = T.argmax(
        T.sum(probs.dimshuffle('x',0,1,2) * weight_samples.dimshuffle(0,1,'x',2), axis=1),
        axis=2)
    accs = T.mean(T.eq(preds, targets.dimshuffle('x',0)), axis=1)
    best_index = T.argmax(accs)
    best_acc = accs[best_index]
    best_weights = weight_samples[best_index]
    print 'compiling functtion'
    fn = theano.function([weight_samples, probs, targets],
        [best_weights, best_index, best_acc])
    print 'done'
    weight_samples_np = numpy_rng.rand(n_iter, n_modal, n_cls).astype(np.float32)
    weight_samples_np /= weight_samples_np.sum(1)[:, None, :]
    
    return fn(weight_samples_np, val_probs, target_val)
