
def interpret_belief_vector(belief_vector, variable_list):
  """
  This function takes in a belief vector and the corresponding variables and
    returns a string interpretation of the belief vector.
  
  The idea is that the function checks the sign of the belief vector and 
    returns a string interpretation of that sign, and prints the value of the belief.

  Example output: "EQWLTH = 3. Believes Government should reduce income differences."
  """

    
  variables = ["PARTYID","POLVIEWS","NATSPAC","NATENVIR","NATHEAL","NATCITY","NATCRIME","NATDRUG","NATEDUC","NATRACE","NATARMS",
"NATAID","NATFARE","NATROAD","NATSOC","NATMASS","NATPARK","NATCHLD","NATSCI","EQWLTH","SPKATH","COLATH","LIBATH","SPKRAC","COLRAC","LIBRAC","SPKCOM","COLCOM","LIBCOM","SPKMIL","COLMIL","LIBMIL","SPKHOMO",
"COLHOMO","LIBHOMO","SPKMSLM","COLMSLM","LIBMSLM","CAPPUN","GUNLAW","COURTS","GRASS","ATTEND","RELITEN","POSTLIFE","PRAYER","AFFRMACT","WRKWAYUP","HELPFUL",
"FAIR","TRUST","CONFINAN","CONBUS","CONCLERG","CONEDUC","CONFED","CONLABOR","CONPRESS","CONMEDIC","CONTV","CONJUDGE","CONSCI","CONLEGIS","CONARMY","GETAHEAD","FEPOL","ABDEFECT","ABNOMORE","ABHLTH","ABPOOR","ABRAPE","ABSINGLE","ABANY","SEXEDUC","DIVLAW","PREMARSX","TEENSEX","XMARSEX","HOMOSEX","PORNLAW",
"SPANKING","LETDIE1","SUICIDE1","SUICIDE2","POLHITOK","POLABUSE","POLMURDR","POLESCAP","POLATTAK","NEWS","TVHOURS","FECHLD","FEPRESCH","FEFAM","RACDIF1","RACDIF2","RACDIF3",
"RACDIF4","HELPPOOR","MARHOMO", "PRESLAST_NONCONFORM", "PRESLAST_DEMREP", "VOTELAST"]

  interpretations = []

  mappings = {
    "PARTYID": {"-": "Leans towards the Democratic party.", "+": "Leans towards the Republican party."},
    "POLVIEWS": {"-": "Is liberal.", "+": "Is conservative."},
    "NATSPAC": {"-": "Believes the government is spending too little on the space exploration program.", "+": "Believes the government is spending too much on the space exploration program."},
    "NATENVIR": {"-": "Believes the government is spending too little on improving and protecting the environment.", "+": "Believes the government is spending too much on improving and protecting the environment."},
    "NATHEAL": {"-": "Believes the government is spending too little on improving and protecting the nation's health.", "+": "Believes the government is spending too much on improving and protecting the nation's health."},
    "NATCITY": {"-": "Believes the government is spending too little on solving the problems of big cities.", "+": "Believes the government is spending too much on solving the problems of big cities."},
    "NATCRIME": {"-": "Believes the government is spending too little on halting the rising crime rate.", "+": "Believes the government is spending too much on halting the rising crime rate."},
    "NATDRUG": {"-": "Believes the government is spending too little on dealing with drug addiction.", "+": "Believes the government is spending too much on dealing with drug addiction."},
    "NATEDUC": {"-": "Believes the government is spending too little on improving the nation's education system.", "+": "Believes the government is spending too much on improving the nation's education system."},
    "NATRACE": {"-": "Believes the government is spending too little on improving the conditions of Blacks.", "+": "Believes the government is spending too much on improving the conditions of Blacks."},
    "NATARMS": {"-": "Believes the government is spending too little on the military, armaments, and defense.", "+": "Believes the government is spending too much on the military, armaments, and defense."},
    "NATFARE": {"-": "Believes the government is spending too little on welfare.", "+": "Believes the government is spending too much on welfare."},
    "NATROAD": {"-": "Believes the government is spending too little on highways and bridges.", "+": "Believes the government is spending too much on highways and bridges."},
    "NATSOC": {"-": "Believes the government is spending too little on Social Security.", "+": "Believes the government is spending too much on Social Security."},
    "NATMASS": {"-": "Believes the government is spending too little on mass transportation.", "+": "Believes the government is spending too much on mass transportation."},
    "NATPARK": {"-": "Believes the government is spending too little on parks and recreation.", "+": "Believes the government is spending too much on parks and recreation."},
    "NATCHLD": {"-": "Believes the government is spending too little on assistance for childcare.", "+": "Believes the government is spending too much on assistance for childcare."},
    "NATSCI": {"-": "Believes the government is spending too little on supporting scientific research.", "+": "Believes the government is spending too much on supporting scientific research."},
    "NATENRGY": {"-": "Believes the government is spending too little on developing alternative energy sources.", "+": "Believes the government is spending too much on developing alternative energy sources."},
    "EQWLTH": {"-": "Believes the government not concern itself with reducing income differences.", "+": "Believes the government should reduce income differences."},
    "SPKATH": {"-": "Believes a person should not be allowed to make a speech against all churches and religions.", "+": "Believes a person should be allowed to make a speech against all churches and religions."},
    "COLATH": {"-": "Believes a person who is against all churches and religions should not be allowed to teach in a college or university.", "+": "Believes a person who is against all churches and religions should be allowed to teach in a college or university"},
    "LIBATH": {"-": "Would not favour the removal of a book against churches and religion from a public library.", "+": "Would favour the removal of a book against churches and religion from a public library."}, 
    "SPKRAC": {"-": "Believes a person should not be allowed to make a speech claiming Blacks are genetically inferior.", "+": "Believes a person should be allowed to make a speech claiming Blacks are genetically inferior."},
    "COLRAC": {"-": "Believes a person who believes Blacks are genetically inferior should not be allowed to teach in a college or university.", "+": "Believes a person who believes Blacks are genetically inferior should be allowed to teach in a college or university."},
    "LIBRAC": {"-": "Would not favour the removal of a book claiming Blacks are genetically inferior from a public library.", "+": "Would not favour the removal of a book claiming Blacks are genetically inferior from a public library."},
    "SPKCOM": {"-": "Believes a communist should not be allowed to make a speech their community.", "+": "Believes a communist should be allowed to make a speech their community."},
    "COLCOM": {"-": "Believes a communist should not be fired from his college teaching position.", "+": "Believes a communist should be fired from his college teaching position."},
    "LIBCOM": {"-": "Would not favour the removal of a book written by a communist from a public library.", "+": "Would favour the removal of a book written by a communist from a public library."},
    "SPKMIL": {"-": "Believes a person who advocates the military running the country, without elections, should not be allowed to speak in their community.", "+": "Believes a person who advocates the military running the country, without elections, should be allowed to speak in their community."},
    "COLMIL": {"-": "Believes a person who advocates the military running the country, without elections, should not be allowed to teach in a college or university.", "+": "Believes a person who advocates the military running the country, without elections, should be allowed to teach in a college or university."},
    "LIBMIL": {"-": "Would not favour the removal of a book advocating the military running the country, without elections, from a public library.", "+": "Would favour the removal of a book advocating the military running the country, without elections, from a public library."},
    "SPKHOMO": {"-": "Believes a homosexual man should not be allowed to make a speech in their community.", "+": "Believes a homosexual man should be allowed to make a speech in their community."},
    "COLHOMO": {"-": "Believes a homosexual man should not be allowed to teach in a college or university.", "+" : "Believes a homosexual man should be allowed to teach in a college or university."},
    "LIBHOMO": {"-": "Would not favour the removal of a book in favour of homosexuality from a public library.", "+": "Would favour the removal of a book in favour of homosexuality from a public library."},
    "SPKMSLM": {"-": "Believes a Muslim clergyman who preaches hatred of the United States should not be allowed to make a speech in their community.", "+": "Believes a Muslim clergyman who preaches hatred of the United States should be allowed to make a speech in their community."},
    "COLMSLM": {"-": "Believes a Muslim clergyman who preaches hatred of the United States should not be allowed to teach in a college or university.", "+": "Believes a Muslim clergyman who preaches hatred of the United States should be allowed to teach in a college or university."},
    "LIBMSLM": {"-": "Would not favour the removal of a book preaching hatred of the United States from a public library.", "+": "Would favour the removal of a book preaching hatred of the United States from a public library."},
    
    "CAPPUN": {"-": "Opposes the death penalty for persons convicted of murder.", "+": "Favours the death penalty for persons convicted of murder."},
    "GUNLAW": {"-": "Opposes a law requiring people to obtain police permits before purchasing a gun", "+": "Favours a law requiring people to obtain police permits before purchasing a gun"},
    "COURTS": {"-": "Believes the courts are too harsh on criminals", "+": "Believes the courts are not harsh enough on criminals"},
    "GRASS": {"-": "Believes marijuana should not be legalised", "+": "Believes marijuana should be legalised"},
    "ATTEND": {"-": "Rarely attends church", "+": "Attends church regularly"},
    "RELITEN": {"-": "Is not religious.", "+": "Is strongly religious."},
    "POSTLIFE": {"-": "Does not believe in life after death.", "+": "Believes in life after death."},
    "PRAYER": {"-": "Dissapproves government mandated readings of Lord's Prayer or Bible verses in public schools.", "+": "Approves government mandated readings of Lord's Prayer or Bible verses in public schools."},
    "AFFRMACT": {"-": "Opposes affirmative action programs for Blacks.", "+": "Favours affirmative action programs for Blacks."},
    "WRKWAYUP": {"-": "Does not believe Irish, Italian, Jewish, and many other minorities overcame predudice and Blacks should do the same without special favours.", "+": "Believes Irish, Italian, Jewish, and many other minorities overcame predudice and Blacks should do the same without special favours."},
    "HELPFUL": {"-": "Believes people look out for themselves most of the time.", "+": "Believes most people try to be helpful most of the time."},
    "FAIR": {"-": "Believes most people would try to take advantage of you, if they got the chance.", "+": "Believes most people would try to be fair with you."},
    "TRUST": {"-": "Believes you can't be too careful when trusting people (people are generally not trustworthy).", "+": "Believes people can generally be trusted."},

    "CONFINAN": {"-": "Has hardly any confidence in the people running banks and financial institutions.", "+": "Has a great deal of confidence in the people running banks and financial institutions."},
    "CONBUS": {"-": "Has hardly any confidence in the people running major companies.", "+": "Has a great deal of confidence in the people running major companies."},
    "CONCLERG": {"-": "Has hardly any confidence in the people running churches and religious organisations.", "+": "Has a great deal of confidence in the people running churches and religious organisations."},
    "CONEDUC": {"-": "Has hardly any confidence in the people running education.", "+": "Has a great deal of confidence in the people running education."},
    "CONFED": {"-": "Has hardly any confidence in the people running the executive branch of the federal government.", "+": "Has a great deal of confidence in the people running the executive branch of the federal government."},

    "CONLABOR": {"-": "Has hardly any confidence in the people running labour organisations (e.g., unions).", "+": "Has a great deal of confidence in the people running labour organisations (e.g., unions)."}, 
    "CONPRESS": {"-": "Has hardly any confidence in the people running the press.", "+": "Has a great deal of confidence in the people running the press."},
    "CONMEDIC": {"-": "Has hardly any confidence in the people running the medical system.", "+": "Has a great deal of confidence in the people running the medical system."},
    "CONTV": {"-": "Has hardly any confidence in the people running television institutions.", "+": "Has a great deal of confidence in the people running television institutions."},
    "CONJUDGE": {"-": "Has hardly any confidence in the people running the U.S. Supreme Court.", "+": "Has a great deal of confidence in the people running the U.S. Supreme Court."},
    "CONSCI": {"-": "Has hardly any confidence in the people running the scientific community.", "+": "Has a great deal of confidence in the people running the scientific community."},
    "CONLEGIS": {"-": "Has hardly any confidence in the people running the U.S. Congress.", "+": "Has a great deal of confidence in the people running the U.S. Congress."},
    "CONARMY": {"-": "Has hardly any confidence in the people running the military.", "+": "Has a great deal of confidence in the people running the military."},

    "GETAHEAD": {"-": "Believes that hard work is more important than luck for getting ahead in life.", "+": "Believes that luck is more important than hard work for getting ahead in life."},
    "FEPOL": {"-": "Disagrees with the idea that most men are better suited emotionally for politics than most women.", "+": "Believes that most men are better suitec for politics than most women."}, 

    "ABDEFECT": {"-": "Believes women should not be able to legally obtain an abortion even if there is a strong chance of serious health defects in the baby.", "+": "Believes that women should be able to legally obtain an abortion if there is a strong chance of serious health defects in the baby."},
    "ABNOMORE": {"-": "Believes women should not be able to legally obtain an abortion even if she is married and does not want any more children.", "+": "Believes women should be able to legally obtain an abortion if she is married and does not want any more children."},
    "ABHLTH": {"-": "Believes women should not be able to legally obtain an abortion even if her health is seriously endangered by the pregnancy.", "+": "Believes women should be able to legally obtain an abortion if her health is seriously endangered by the pregnancy."},
    "ABPOOR": {"-": "Believes women should not be able to legally obtain an abortion even if the family has very low income and cannot afford more children.", "+" : "Believes women should be able to legally obtain an abortion if the family has very low income and cannot afford more children."},
    "ABRAPE": {"-": "Believes women should not be able to legally obtain an abortion even if she became pregnant as a result of rape.", "+": "Believes women should be able to legally obtain an abortion if she became pregnant as a result of rape."},

    "ABSINGLE": {"-": "Believes women should not be able to legally obtain an abortion even if she is not married and does not want to marry the father.", "+": "Believes women should be able to legally obtain an abortion if she is not married and does not want to marry the father."},
    "ABANY": {"-": "Believes women should not be able to legally obtain an abortion for any given reason.", "+" : "Believes women should be able to legally obtain an abortion for any reason they wants."},

    "SEXEDUC": {"-": "Opposes sex education in public schools.", "+": "Favours sex education in public schools."},
    "DIVLAW": {"-": "Believes that divorce in this country should more difficult to obtain than it is now.", "+": "Believes that divorce in this country should be easier to obtain than it is now."},


    "PREMARSX": {"-": "Believes that premarital sex is not wrong at all.", "+": "Believes that premarital sex is always wrong."},
    "TEENSEX": {"-": "Believes that teenagers having sexual relations before marriage is not wrong at all.", "+": "Believes that teenagers having sexual relations before marriage is always wrong."},
    "XMARSEX": {"-": "Belives that extramarital sex is not wrong at all.", "+": "Believes that extramarital sex is always wrong."},

    "HOMOSEX": {"-": "Believes that sex between two adults of the same sex is not wrong at all.", "+": "Believes that sex between two adults of the same sex is always wrong."},
    "PORNLAW": {"-": "Believes that there should be no laws forbidding the distribution of pornography.", "+": "Believes that there should be laws forbidding the distribution of pornography."},
    "SPANKING": {"-": "Believes that it is never necessary to spank a child.", "+": "Believes that it is sometimes necessary to spank a child."},
    "LETDIE1": {"-": "Believes that a doctor should never be allowed to painlessly end the life of a patient with an incurable disease if the family requests it.", "+": "Believes that a doctor should be allowed to painlessly end the life of a patient with an incurable disease if the family requests it."},
    "SUICIDE1": {"-": "Believes that a person should never be allowed to end their life if they have an incurable disease.", "+": "Believes that a person should be allowed to end their life if they have an incurable disease."},
    "SUICIDE2": {"-": "Believes that a person should never be allowed to end their life if they have gone bankrupt.", "+": "Believes that a person should be allowed to end their life if they have gone bankrupt."},
    
    "POLHITOK": {"-": "Believes there are no situations where it is okay for a policeman to strike an adult male citizen.", "+": "Believes there are situations where it is okay for a policeman to strike an adult male citizen."},
    "POLABUSE": {"-": "Believes that it is not okay for a policeman to strike an adult male citizen who has said vulgar and obscene things to the policeman.", "+": "Believes that it is okay for a policeman to strike an adult male citizen who has said vulgar and obscene things to the policeman."},
    "POLMURDR": {"-": "Believes that it is not okay for a policeman to strike an adult male citizen who is being questioned as a suspect in a murder case.", "+": "Believes that it is okay for a policeman to strike an adult male citizen who is being questioned as a suspect in a murder case."},
    "POLESCAP": {"-": "Believes that it is not okay for a policeman to strike an adult male citizen who is attempting to escape from custody.", "+": "Believes that it is okay for a policeman to strike an adult male citizen who is attempting to escape from custody."},
    "POLATTAK": {"-": "Believes that it is not okay for a policeman to strike an adult male citizen who is attacking him.", "+": "Believes that it is okay for a policeman to strike an adult male citizen who is attacking him."},

    "NEWS": {"-": "Hardly ever reads the newspaper .", "+": "Reads the newspaper every day."}, 
    "TVHOURS": {"-": "Hardly ever watches TV.", "+": "Watches TV all the time."},
    "FECHLD": {"-": "Believes a working mother cannot establish a relationship that is as warm and secure with her children as a mother who does not work.", "+": "Believes a working mother can establish a relationship that is as warm and secure with her children as a mother who does not work."},
    "FEPRESCH": {"-": "Believes that a preschool child is not likely to suffer if their mother works.", "+": "Believes that a preschool child is likely to suffer if their mother works."},
    "FEFAM": {"-": "Believes that it is not better for everyone if the man is the achiever outside the home and the woman takes care of the home and family.", "+": "Believes that it is better for everyone if the man is the achiever outside the home and the woman takes care of the home and family."},

    "RACDIF1": {"-": "Does not believe the main cause for Blacks having worse jobs, income, and housing than white people is discrimination.", "+": "Believes the main cause for Blacks having worse jobs, income, and housing than white people is discrimination."},
    "RACDIF2": {"-": "Does not believe the main cause for Blacks having worse jobs, income, and housing than white people is that most Blacks have less in-born ability to learn.", "+": "Believes the main cause for Blacks having worse jobs, income, and housing than white people is that most Blacks have less in-born ability to learn."},
    "RACDIF3": {"-": "Does not believe the main cause for Blacks having worse jobs, income, and housing than white people is that most Blacks don't have the chance for education that it takes to rise out of poverty.", "+": "Believes the main cause for Blacks having worse jobs, income, and housing than white people is that most Blacks don't have the chance for education that it takes to rise out of poverty."},
    "RACDIF4": {"-": "Does not believe the main cause for Blacks having worse jobs, income, and housing than white people is that most Blacks don't have themotivation or will power to pull themselves out of poverty.", "+": "Believes the main cause for Blacks having worse jobs, income, and housing than white people is that most Blacks don't have the motivation or will power to pull themselves out of poverty."},

    "HELPPOOR": {"-": "Does not believe that it is the responsibility of the government to improve the living stantard of all poor Americans.", "+": "Believes that it is the responsibility of the government to improve the living stantard of all poor Americans."},
    "MARHOMO": {"-": "Believes that it is not okay for homosexuals to marry.", "+": "Believes that it is okay for homosexuals to marry."},

    "PRESLAST_NONCONFORM": {"-": "Voted for a major party in the last major election.", "+": "Voted for a third party in the last major election."},
    "PRESLAST_DEMREP": {"-": "Voted for a Democrat in the last major election.", "+": "Voted for a Republican in the last major election."},
    "VOTELAST": {"-": "Did not vote in the last major election.", "+": "Voted in the last major election."}
    }     


  for belief, variable in zip(belief_vector, variable_list):
      if variable in mappings:
          if belief > 0:
              interpretation = f"{variable} = {belief}. {mappings[variable]['+']}"
          elif belief < 0:
              interpretation = f"{variable} = {belief}. {mappings[variable]['-']}"
          else:
              interpretation = f"{variable} = {belief}. Neutral."
          interpretations.append(interpretation)
      else:
          interpretations.append(f"{variable} = {belief}. Unknown variable. No intepretation available.")
  
  return "\n".join(interpretations)




