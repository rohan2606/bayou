import re
from nltk.stem.wordnet import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

def split_words_underscore_plus_camel(s):
    s = re.sub('_', '#', s)
    s = re.sub('(.)([A-Z][a-z]+)', r'\1#\2', s)  # UC followed by LC
    s = re.sub('([a-z0-9])([A-Z])', r'\1#\2', s)  # LC followed by UC
    vars = s.split('#')

    final_vars = []
    for var in vars:
        var = var.lower()
        w = lemmatizer.lemmatize(var, 'v')
        w = lemmatizer.lemmatize(w, 'n')
        if len(w) > 1:
            final_vars.append(w)
    return final_vars

def get_variables(body):

    lines = body.split("\n")

    variable_names = []
    header_variable_names = []

    countLines = 0
    for line in lines:

        # detect first line!
        if '*' in line:
            continue
        countLines += 1

        line = line.replace("final","")

        line = re.sub(r'@\S*', '', line).strip() #line.replace("@.* ","")
        line = line.replace("==","")
        line = line.replace("!=","")
        line = line.replace(">=","")
        line = line.replace("<=","")

        line = line.strip()
        # print(line)
        if(re.search( "{$" , line) and (countLines==1) ):

            input = re.findall('\([a-zA-Z0-9 ,<>_\[\]\.?@]*\)',line)
            if len(input) == 0:
                continue
            else:
                input = input[0]
            # print(input)
            #remove the brackets
            input = input[1:-1]
            if len(input) > 0:
                params = input.strip().split(",")
                for param in params:
                    variable_name = param.strip().split(" ")
                    if len(variable_name)==1:
                        continue
                    else:
                        variable_name = variable_name[1]
                        header_variable_names.append(variable_name)

        elif ('=' in line ):

            lhs = line.split("=")[0]
            declaration = lhs.strip().split(" ")

            if len(declaration) == 2: # standard initialization
                variable_name = declaration[1]
                variable_name = re.findall(r'\w+', variable_name) # strip alphanumeric like += , -= etc
                if(len(variable_name)>0):
                    variable_name = variable_name[0]
                    variable_names.append(variable_name)

            elif len(declaration) == 1: # dot assignment or increments etc
                vars = declaration[0].split('.')

                for var in vars:

                    var = re.findall(r'\w+', var)# strip alphanumeric like += , -= etc
                    if(len(var)>0):
                        var = var[0]
                        variable_names.append(var)

        elif ( re.search( ";$" , line) and len(line.strip().split(" "))==2 and ('return' not in line) and ('.' not in line) and ('(' not in line) and (')' not in line) ):

            variable_name = line.strip().split(" ")[1]
            variable_name = variable_name[:-1] # removing the ;
            variable_names.append(variable_name)

    filtered_variables = list(set(variable_names) - set(['this']))
    final_vars = []
    for var in filtered_variables:
        if len(var)>0:
            var = re.sub("\d", "", var) #remove digits
            split_vars = split_words_underscore_plus_camel(var)
            final_vars.extend(split_vars)

    return header_variable_names, list(set(final_vars)),
