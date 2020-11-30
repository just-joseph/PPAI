import time
start= time.time()

import os
import re
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

# from remove_duplicates import *
import collections
import glob
import pandas as pd
import time
# import uuid
import sys

def txt_to_csv(txt, outputfolder, job_id):
    # print("Entered txt to csv")
    fileExtension = txt.split(".")[-1]
    if fileExtension == "txt":
        with open(txt,  encoding="latin-1") as f:
            print("txtFilename = ", txt)
            lines=f.read()
        tests = lines.split("\n\n")
        df= pd.DataFrame()
        df['Paragraph'] = pd.Series(tests)
        df.to_csv(outputfolder+txt+"_"+job_id+".csv")
            # , encoding='latin-1')

def create_tokenizer_score(new_series, train_series, tokenizer):
    """
    return the tf idf score of each possible pairs of documents
    Args:
        new_series (pd.Series): new data (To compare against train data)
        train_series (pd.Series): train data (To fit the tf-idf transformer)
    Returns:
        pd.DataFrame
    """
    train_tfidf = tokenizer.fit_transform(train_series)
    new_tfidf = tokenizer.transform(new_series)
    score=cosine_similarity(new_tfidf, train_tfidf)[0]
    return score
    
def similar_para(pdf_df,filename, clause):
    df_C = pdf_df.copy()
    df_C =  df_C.replace(np.nan, '', regex=True)
    df_C['FileName'] = filename.split('.txt')[0]
    train_set = df_C['Paragraph']
    df_C['Clause'] = clause
    test_set = pd.Series(clause)
    tokenizer = TfidfVectorizer()
    score = create_tokenizer_score(train_series = train_set, new_series = test_set, tokenizer = tokenizer)
    df_C['Similarity score'] = score
    return df_C

def model_pipeline(context,question, model_path):
    nlp_qa = pipeline('question-answering',model = model_path, tokenizer = model_path)
        # , device=0)
    output = nlp_qa(context=context, question=question,topk =2)
    bert_output = pd.DataFrame(output)
    return bert_output
    
def match_para_start(booldf,bert_output):
    bool_list = list()
    for rows in bert_output.iterrows():
        start_ans = rows[1]['start']
        for bool_row in booldf.to_dict(orient ="row"):
            if bool_row['start']<start_ans and bool_row['end']>start_ans:
                bool_row['Answer'] = rows[1]['answer']
                bool_list.append(bool_row)
    matched_df = pd.DataFrame(bool_list)
    return matched_df

def main(job_id, textfile, clause_rule_csv_path):
	try:
		# input_folder= "./predict_files/Input_files/input_txt/"
		# textfile= textfile
		# clause_rule_csv= clause_rule_csv
		txt_to_csv_folder= "./predict_files/inter_files/txt_Data/"
		model_path= './predict_files/model_files'
		output_folder= "./predict_files/Output/"
		log_folder= "./log/"
		inter_folder= "./predict_files/inter_files/"

		logfile = open(log_folder+"Logfile_"+job_id+".txt","w", encoding="latin-1")#write mode 
		logfile.write("Status: Loaded packages\n") 
		logfile.write("Time: Loading packages completed by "+ str(time.time()-start)+"\n")
		logfile.flush()

		txt_to_csv(textfile, txt_to_csv_folder, job_id)
		# print("txt to csv converted")
		logfile.write("Status: txt converted to paragraphs csv\n") 
		logfile.write("Time: txt converted to paragraphs csv by "+ str(time.time()-start)+"\n")
		logfile.flush()

		# print(clause_rule_csv_path)
		clauses_df= pd.read_excel(clause_rule_csv_path)
		# clauses_dict= pd.Series(clauses_df.Clause.values,index= clauses_df.Clause_name).to_dict()
		# clauses_ordered_dict= collections.OrderedDict(clauses_dict)
		# list_clauses_ordered_dict= list( clauses_ordered_dict.items() )

		# Use this if we want to look into all files in txt_to_csv_folder
		# csvfiles = glob.glob(os.path.join(txt_to_csv_folder, '*.csv'))

		# Use this to look into only one file in txt_to_csv_folder
		csvfiles=[]
		for file in os.listdir(txt_to_csv_folder):
			if(file.endswith(job_id+'.csv')):
				csvfiles.append(txt_to_csv_folder+file)
		print(" CSVFILES ", csvfiles)

		final_list=[]
		# print("processes till iteration of files completed by ", time.time()- start)

		for files in csvfiles:
			csv_data = pd.read_csv(files)
			sim_df_per_file= pd.DataFrame()
			file_name= files.split("/")[-1]
			# print("File name is ", file_name)
			attribute_values= {'Filename':file_name.split('.txt')[0]}

			logfile.write("\nStatus: Triggering clause similarity calculation for file w.r.t each clause\n")
			logfile.flush()

			for index, row in clauses_df.iterrows():
				print("Starting ", row['Clause_name'])
				logfile.write("\nStarting processing of "+ row['Clause_name'])
				logfile.flush()
				# print(row['Clause'])

				new_df = similar_para(csv_data, file_name, row['Clause'])

				top_sim = new_df.sort_values(by=['Similarity score'], ascending=False).head(5)
				# print(top_sim)
				sim_df_per_file= sim_df_per_file.append(top_sim, ignore_index=True)
				logfile.write("\n"+"Time: Clause similarity calculated for file completed by "+ str(time.time()-start)+"\n")
				logfile.flush()

				if( str(row['Similarity_threshold'])!= 'nan' ):
					print("entering standard vs negotiated check")
					if (top_sim['Similarity score'].iloc[0]< row['Similarity_threshold']):
						logfile.write(row['Clause_name']+" is negotiated\n")
						if( str(row['Enter_QA_only_if_negotiated'])=='Yes' and str(row['Question'])!='nan' ):
							print("Entered negotiated QA")
							print("entering QA check with ", str(row['Question']))
							bert_output = model_pipeline( top_sim.Paragraph.str.cat(sep=' '), str(row['Question']), model_path)
							attribute_values[row['Clause_name']]= bert_output['answer'].iloc[0]
							logfile.write("QA model ran for "+ row['Clause_name']+ " attribute\n") 
							logfile.write("Time: QA model ran for "+ row['Clause_name']+ " attribute by"+ str(time.time()-start)+"\n")
							logfile.flush()
						else:
							attribute_values[ row['Clause_name'] ]= 'Negotiated : '+ top_sim['Paragraph'].iloc[0]
					else:
						logfile.write(row['Clause_name']+" is standard\n")
						if( str(row['Enter_QA_only_if_negotiated'])=='No' and str(row['Question'])!='nan' ):
							print("Entered standard QA")
							print("entering QA check with ", str(row['Question']))
							bert_output = model_pipeline( top_sim.Paragraph.str.cat(sep=' '), str(row['Question']), model_path)
							attribute_values[row['Clause_name']]= bert_output['answer'].iloc[0]
							logfile.write("QA model ran for "+ row['Clause_name']+ " attribute\n") 
							logfile.write("Time: QA model ran for "+ row['Clause_name']+ " attribute by"+ str(time.time()-start)+"\n")
							logfile.flush()
						else:
							attribute_values[ row['Clause_name'] ]= 'Standard'
				else:	# ENTER_QA_ONLY)IF_NEGOTIATED HAS NO RELEVANCE WHEN SIMILARITY THRESHOLD NOT PRESENT i.e. WE ARE NOT CHECKING IF STANDARD OR NEGOTIATED
					print("not entering standard vs negotiated check as similarity threshold not present")
					if( str(row['Question'])!='nan' ):
						print("Entered QA without any standard or negotiated check")
						print("entering QA check with ", str(row['Question']))
						bert_output = model_pipeline( top_sim.Paragraph.str.cat(sep=' '), str(row['Question']), model_path)
						attribute_values[row['Clause_name']]= bert_output['answer'].iloc[0]
						logfile.write("QA model ran for "+ row['Clause_name']+ " attribute\n") 
						logfile.write("Time: QA model ran for "+ row['Clause_name']+ " attribute by"+ str(time.time()-start)+"\n")
						logfile.flush()

				logfile.write("Status: "+ row['Clause_name']+" clause has finished examination\n")
				logfile.flush()
				print("Completed clause")

			# print("attribute values are ", attribute_values)
			logfile.write("\nFinding: attribute values are "+ str(attribute_values)+"\n")
			final_list.append(attribute_values)
			# print(sim_df_per_file)
			sim_df_per_file.to_csv(inter_folder+"Top_similarity_"+file_name+".csv", index=False)
			# print("File processing completed by ", time.time()- start)
			logfile.write("File processing completed by "+ str(time.time()-start) +"\n")

			pd.DataFrame(final_list).to_csv(output_folder+"Final_output_"+job_id+".csv", index=False)
			logfile.write("\nStatus: Processing completed")
			logfile.flush()
			logfile.close()
			print("Completed file")
	except Exception as e:
		print("Error! ", e)
		logfile.write("\n\nError: ", e)
		logfile.flush()
		logfile.close()


if __name__ == '__main__':
	job_id= sys.argv[1]
	# txtfile= sys.argv[2]
	txtfile = sys.argv[2]
	clause_rule_csv= sys.argv[3]
	# print(job_id)
	# print(txtfile_path)
	# print(clause_rule_csv)
	# print(type(clause_rule_csv))
	main(job_id, txtfile, clause_rule_csv)
