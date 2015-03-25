import numpy as np
import time
from operator import add

from toolbox import *
from models import *

if __name__ == "__main__":

#   Hyper-parameters
#--------------------------------------------------------------------------------------------------
    hp = Parameters()
    with hp:
        batch_size = 1000
        test_batch_size = 1000
        train_perm = False

        load_model = False
        save_model = True

        walkforward = True
        walkstep_size = 5
        n_iterations = 100
        n_iterdecay = 0.90

        debug = False
        resample_z = False

        #Model = ffn.FFN
        #Model = cnn.CNN
        
        #Model = vae1.Vae1
        #Model = cvae.Cvae 
        #Model = draw_at_lstm1.Draw_at_lstm1 
        #Model = draw_at_lstm2.Draw_at_lstm2 
        Model = draw_lstm1.Draw_lstm1 
        #Model = draw_sgru1.Draw_sgru1 

        init_scale = 1.3
        learning_rate = 0.0008

        ''' sgd(1),  rmsprop(0.001),  adam(0.0005),  adamgc(0.0005),  esgd(0.01) '''
        optimizer = adamgc


#   Data
#--------------------------------------------------------------------------------------------------
    data_path = 'data/'

    #data = mnist(path=data_path, nvalidation=0) 
    data = mnistBinarized(path=data_path)  # only for UL models

    #data = mnist(path=data_path, distort=3, shuffle=True) 
    #data = freyfaces(path=data_path)

    visualize(-1, data['tr_X'][0:min(len(data['tr_X']), 900)], data['shape_x'])


#   Training
#--------------------------------------------------------------------------------------------------
    model = Model(data, hp)
    
    print ("M: %s  lr: %.5f  init: %.1f  batch: %d  ws: %d  iter: %d" % (model.id, learning_rate, init_scale, batch_size, walkforward*walkstep_size, n_iterations)) 
    
    if walkforward:
        # Walkforward learning
        if model.type == 'SL':
            # Supervised learning
            n_ws = len(data['tr_X']) / walkstep_size / batch_size
            it_lr = learning_rate
            for walkstep in xrange(0, n_ws):
                begin = time.time()
                #it_lr = float(learning_rate / (walkstep + 1.))
                classRateTr = 0.
                for it in range(n_iterations):
                    for i in xrange(0, walkstep_size):
                        batch_idx = walkstep * walkstep_size + i
                        predictX = model.train(batch_idx, it_lr)
                        if it==0:
                            classRateTr += np.sum(np.argmax(data['tr_Y'][batch_idx*batch_size:(batch_idx+1)*batch_size], axis=1) == predictX)
                
                #it_lr = it_lr*n_iterdecay
                n_iterations = int(n_iterations*n_iterdecay)
                wf_error_rate = 1.0 - classRateTr / walkstep_size / batch_size
                test_error_rate = 1.0 - np.mean(np.argmax(data['te_Y'], axis=1) == model.test())
                print("%d,%.4f,%.4f,%.4f,%.2f" % (walkstep + 1, it_lr, wf_error_rate, test_error_rate, time.time() - begin))
       
        elif model.type == 'UL':
            # Unsupervised learning
            n_ws = len(data['tr_X']) / walkstep_size / batch_size
            it_lr = learning_rate
            for walkstep in xrange(0, n_ws):
                begin = time.time()
                #it_lr = float(learning_rate / (walkstep + 1.))
                tr_cost = (0., 0.)
                for it in range(n_iterations):
                    for i in xrange(0, walkstep_size):
                        batch_idx = walkstep * walkstep_size + i
                        cost_list = model.train(batch_idx, it_lr)
                        if it==0:
                            tr_cost = map(add, tr_cost, cost_list)

                te_cost = (0., 0.)
                for i in xrange(0, len(data['te_X']) / test_batch_size):
                    cost_list = model.testBatch(i)
                    te_cost = map(add, te_cost, cost_list)
                
                #it_lr = it_lr*n_iterdecay
                n_iterations = int(n_iterations*n_iterdecay)
                print("%d,%.4f,%.2f,%.2f,%.2f,%.2f,%.2f" % (walkstep + 1, it_lr, 
                                                       tr_cost[1] / walkstep_size / batch_size, 
                                                       te_cost[1] / len(data['te_X']), 
                                                       np.sum(tr_cost) / walkstep_size / batch_size, 
                                                       np.sum(te_cost) / len(data['te_X']), 
                                                       time.time() - begin))
                # Generate samples
                y_samples = model.decode(36 * (model.n_t + 1))
                y_samples = np.transpose(y_samples, (1,0,2)).reshape((-1, y_samples.shape[2]))
                visualize(walkstep + 1, y_samples, data['shape_x'])

    else:
        # Full training data learning
        n_iterations = 10000
        freq_save = 20
        freq_sample = 10

        if model.type == 'SL':
            # Supervised learning
            for it in range(n_iterations):
                begin = time.time()
                model.permuteData(data)
                classRateTr = 0.

                for i in xrange(0, len(data['tr_X']) / batch_size):
                    predictTrX = model.train(i)
                    classRateTr = classRateTr + np.sum(np.argmax(data['tr_Y'][i*batch_size:(i+1)*batch_size], axis=1) == predictTrX)
    
                end = time.time()
                print("%d,%.4f,%.4f,%.2f" % (it, 1 - classRateTr / len(data['tr_X']), 
                                             1 - np.mean(np.argmax(data['te_Y'], axis=1) == model.test()), 
                                             end - begin))
                # Save model parameters
                if hp.save_model and it % freq_save == 0:
                    model.save()

        elif model.type == 'UL':
            # Unsupervised learning
            for it in range(n_iterations):
                begin = time.time()
                model.permuteData()
            
                tr_cost_p = 0.
                tr_cost_q = 0.
                for i in xrange(0, len(data['tr_X']) / batch_size):
                    log_p, log_q = model.train(i)
                    tr_cost_p += log_p
                    tr_cost_q += log_q

                te_cost_p = 0.
                te_cost_q = 0.
                for i in xrange(0, len(data['te_X']) / test_batch_size):
                    log_p, log_q = model.testBatch(i)
                    te_cost_p += log_p
                    te_cost_q += log_q

                end = time.time()
                print("%d,%.2f,%.2f,%.2f,%.2f,%.2f" % (it, 
                                                       tr_cost_q / len(data['tr_X']), 
                                                       te_cost_q / len(data['te_X']), 
                                                       (tr_cost_p + tr_cost_q) / len(data['tr_X']),
                                                       (te_cost_p + te_cost_q) / len(data['te_X']),
                                                       end - begin))
                # Generate samples
                if it % freq_sample == 0:
                    y_samples = model.decode(36 * (model.n_t + 1))
                    y_samples = np.transpose(y_samples, (1,0,2)).reshape((-1, y_samples.shape[2]))
                    visualize(it, y_samples, data['shape_x'])
        
                # Save model parameters
                if hp.save_model and it % freq_save == 0:
                    model.save()

