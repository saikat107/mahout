import json
import pandas as pd
import pickle
import ast
import numpy as np
import copy

data = []

with open('25_K_Examples/part-1-output/taken_answers_with_all_details.json', 'r') as f:
	data += json.load(f)

with open('25_K_Examples/part-2-output/taken_answers_with_all_details.json', 'r') as f:
	data += json.load(f)

with open('25_K_Examples/part-3-output/taken_answers_with_all_details.json', 'r') as f:
	data += json.load(f)

def is_string(s):
	try:
		f = float(s)
		return False
	except:
		if s == 'True' or s == 'False':
			return False
		if is_time(s) or is_date(s):
			return False
		return True

def is_date(s):
	ymd = s.split('-')
	if len(ymd) == 3 and all(x.isnumeric() for x in ymd):
		return True
	return False

def is_time(s):
	hms = s.split(':')
	if len(hms) == 3 and all(x.isnumeric() for x in hms):
		return True
	return False

num_success = 0

for i, question in enumerate(data):
	io = question['formatted_input']['io']

	try:
		question['formatted_input']['dfs'] = []

		for example in io:
			lines = example.split('\n')

			lines = [line.strip('[]') for line in lines]

			if '>>>' in lines[0]:
				lines = lines[1:]
			if 'df' in lines[0]:
				lines = lines[1:]

			potential_heading = lines[0].split()
			sep = None

			if len(potential_heading) == 1:
				potential_heading = lines[0].split(',')
				sep = ','

				if len(potential_heading) == 1:
					potential_heading = lines[0].split(';')
					sep = ';'

					if len(potential_heading) == 1:
						raise Exception

			if all(is_string(cell) for cell in potential_heading):
				if not all(is_string(cell) for cell in lines[1]):
					heading = potential_heading
					lines = lines[1:]
				else:
					heading = [str(i) for i in range(len(potential_heading))]
			else:
				heading = [str(i) for i in range(len(potential_heading))]

			heading = [h for h in heading if h.strip() != '...']

			num_cols = len(heading)

			df_data = []

			for line in lines:
				if len(line) == 0:
					continue

				if line.strip() == '...':
					continue

				if '#' in line:
					comm_ind = line.index('#')
					line = line[:comm_ind]

				if sep is not None:
					cells = line.split(sep)
				else:
					cells = line.split()

				cells = [c for c in cells if c.strip() != '...']

				if sep is None:
					new_cells = []
					j = 0
					while j < len(cells):
						if (j < len(cells) - 1) and is_date(cells[j]) and is_time(cells[j+1]):
							cells[j] = cells[j] + ' ' + cells[j+1]
							new_cells.append(cells[j])
							j += 2
						else:
							new_cells.append(cells[j])
							j += 1
					cells = new_cells

				if len(cells) == num_cols:
					row = cells
				elif len(cells) == num_cols + 1:
					row = cells[1:]
				elif len(cells) == 0:
					continue
				else:
					raise Exception

				df_data.append(row)

			df = pd.DataFrame(df_data, columns=heading)

			for key in df:
				if df[key].dtype == 'object':
					try:
						df[key] = df[key].astype(int)
					except:
						try:
							df[key] = df[key].astype(float)
						except:
							continue

			question['formatted_input']['dfs'].append(df)

		num_success += 1

	except:
		pass

print("Succeeded in {} out of {}".format(num_success, len(data)))

# with open('25_K_Examples/part-2-output/answers_with_processed_io.pkl', 'wb') as f:
	# pickle.dump(data, f)

def compare_dfs(df1, df2):

	if set(df1.columns) != set(df2.columns):
		return False

	for key in df1:
		assert (key in df2)

		if df1[key].dtype == 'float64' and df2[key].dtype == 'float64':
			if not np.allclose(df1[key].to_numpy(), df2[key].to_numpy(), equal_nan=True):
				return False
		else:
			if not df1[key].equals(df2[key]):
				return False

	return True

successful_questions = []

for i, question in enumerate(data):
	dfs = question['formatted_input']['dfs']

	if len(dfs) != 2:
		continue

	inp, out = dfs

	if inp.equals(out):
		continue

	code_strs = question['formatted_input']['answer']['code']
	success = False

	for code_str in code_strs:

		if success:
			break

		code_str = '\n'.join([line.strip('<>') for line in code_str.split('\n')])

		id_names = []

		try:
			ast_obj = ast.parse(code_str)
		except:
			continue

		for node in ast.walk(ast_obj):
			if isinstance(node, ast.Name):
				id_names.append(node.id)

		code_str = "import numpy as np\nimport pandas as pd\n" + code_str

		for id_name in set(id_names):

			if success:
				break

			inp_copy = inp.copy(deep=True)

			loc = {id_name : inp_copy}

			try:
				exec(code_str, dict(), loc)

				for key, val in loc.items():
					if isinstance(val, pd.DataFrame) and compare_dfs(val, out):
						print("IT WORKED!!!!")
						success = True
						final_dict = {'title' 		: question['formatted_input']['question']['title'],
									  'ques_desc' 	: question['formatted_input']['question']['ques_desc'],
									  'ans_desc' 	: question['formatted_input']['answer']['ans_desc'],
									  'code' 	 	: code_str,
									  'in_id_name'	: id_name,
									  'out_id_name'	: key,
									  'inp_df' 		: inp,
									  'out_df' 		: out,
									 }
						successful_questions += [final_dict]
						break

				del inp_copy
			except:
				# import pdb; pdb.set_trace()
				del inp_copy
				continue

with open('successful_questions.pkl', 'wb') as f:
	pickle.dump(successful_questions, f)

import pdb; pdb.set_trace()
