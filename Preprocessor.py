import numpy as np
import pandas as pd
import soundfile as sf

# Class for input and output preprocessing
class Preprocessor:
    
    # returns the numpy array representing Speech and Non speech value where
    # chunk size => millis (default value of millis = 20 ms)
    def preprocess_output_file(self, filename, millis = 20):
        data = pd.read_csv(filename)
        #print(data)
        
        # Converting seconds into millis
        data["tmi0"] = data["tmi0"] * 1000
        data["tmax"] = data["tmax"] * 1000
        #print(data)
        
        start_val = 0.0
        result = np.empty((0,1))
        
        for index, row in data.iterrows():
            
            # row["text"]: 1 if speech and 0 if non-speech
            state = row["text"]
            
            # complete chunks
            diff = row["tmax"] - start_val
            no_of_complete_chunks = int(diff/millis)
            #print(no_of_complete_chunks)
            if(state == 0):
                arr_to_append = np.zeros((no_of_complete_chunks,1))
            elif(state == 1):
                arr_to_append = np.ones((no_of_complete_chunks,1))
                
            result = np.append(result, arr_to_append, axis=0)
                
            # overlapping chunks
            extra = diff - (no_of_complete_chunks * millis)
            if(extra > 0):
                hf = millis/2   # if millis = 20ms the hf is 10
                # if millis = 20ms, extra >= 10ms and currently we are on SPEECH(state=1)
                # then we'll consider appending 1 otherwise 0
                # So, the below condition takes care of assignining proper state value(whether
                # it is speech or non_speech based on the overlapping duration)
                # Speech OR Nonspeech value is assigned considering the major chunk of
                # overlapping data
                if((extra >= hf and state == 0) or (extra < hf and state == 1)):
                    arr_to_append = np.zeros((1,1))
                elif((extra >= hf and state == 1) or (extra < hf and state == 0)):
                    arr_to_append = np.ones((1,1))
                result = np.append(result, arr_to_append, axis=0)
                start_val = start_val + ((no_of_complete_chunks * millis) + millis)
            else:
                start_val = start_val + (no_of_complete_chunks * millis)
            
        return result
    
    def preprocess_input_file(self, filename, channel=1, millis=20):
        data, fs = sf.read(filename)
        if(channel == 1):
            channel_data = data[:,0:1]
        else:
            channel_data = data[:,1:2]
        
        print("Total amount of data in channel "+str(channel), channel_data.shape[0])
        
        # data for 20 ms
        chunk_size = int((fs/1000)*millis)
        print("Size of each data", chunk_size)
        
        itr = int(channel_data.shape[0]/chunk_size)
        print("No. of passes", itr)
        inp = list()
        for i in range(itr):
            start = i * chunk_size
            chunked_data = channel_data[start:(start + chunk_size),:]
            chunk_data_reshaped = np.reshape(chunked_data, chunk_size)
            #print (chunk_data_reshaped.shape)
            inp.append(chunk_data_reshaped)
        return np.array(inp)
