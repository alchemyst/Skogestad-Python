s/[[:blank:]]*$//g
s/ *= */ = /g
s/=  =/==/g
s/\([<>!]\) =/\1=/g
s/,\([^ ]\)/, \1/g
s/return */return /g
