import tensorflow as tf
import bayou.models.low_level_evidences.predict
import argparse
import sys
import socket
import json

class Predictor:

    def __init__(self):
        #set clargs.continue_from = True while testing, it continues from old saved config
        clargs.continue_from = True
        print('Loading Model, please wait _/\_ ...')
        model = bayou.models.low_level_evidences.predict.BayesianPredictor

        sess = tf.InteractiveSession()
        self.predictor = model(clargs.save, sess, batch_size=1)# goes to predict.BayesianPredictor

        print ('Model Loaded, All Ready to Predict Evidences!!')

        return


class Encoder_Model:

    def __init__(self, predictor):
        self.predictor = predictor
        return

    def get_latent_space(self, program):
        _, EncA, EncB = self.predictor.predictor.get_psi_encoder(program)
        return _, EncA, EncB


def socket_server(encoder):
	serv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	serv.bind((socket.gethostname(), 5000))
	serv.listen(5)
	print('Server Starting!')
	while True:
	     conn, addr = serv.accept()
	     print('Client Accepted')
	     #from_client = ''
	     while True:
	         data = conn.recv(1000000)
	         data.decode()
	         if not data: break
	         #from_client += data
	         print(data)
	         _, eA, eB = encoder.get_latent_space(json.loads(data))
	         eA = eA[0].item()
	         eB = [val.item() for val in eB[0]] 	 
	         #send_data='i am server'
	         js = {'eA':eA, 'eB':eB}
	         send_data = json.dumps(js)
	         conn.send(send_data.encode())
	conn.close()
	print ('client disconnected')





if __name__ == "__main__":
    #%%
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--python_recursion_limit', type=int, default=10000,
    help='set recursion limit for the Python interpreter')

    parser.add_argument('--save', type=str, default='/home/ubuntu/save_500_new_drop_skinny_seq_surr_cont')

    clargs = parser.parse_args()
    sys.setrecursionlimit(clargs.python_recursion_limit)


    pred = Predictor()
    encoder = Encoder_Model(pred)
    
    socket_server(encoder)
    

