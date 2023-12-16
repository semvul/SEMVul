import os

dotFiles = "./Vul"
dotlist = os.listdir(dotFiles)
for eachFile in dotlist:
    print(eachFile)
    f=open(dotFiles+"/"+eachFile)
    allFile=f.read()
    lines=allFile.split("\n")
    f.close()
    indexDict={}
    res=""
    for each in lines:
        if each.startswith("\""):
            if each.find("<SUB>")!=-1:
                lineNum = each[each.index("<SUB>") + 5:each.index("</SUB>")]
                for i in range(1,len(each)):
                    if each[i]=="\"":
                        nodeNum=each[1:i]
                        break
                indexDict[nodeNum]=lineNum
                ori = "\"" + nodeNum + "\""
                target = "\"" + lineNum + "\""
                each=each.replace(ori,target)
            else:
                for i in range(1,len(each)):
                    if each[i]=="\"":
                        nodeNum=each[1:i]
                        break
                indexDict[nodeNum]=-1
                continue
        else:
            splitpos=[]
            for i in range(1,len(each)):
                if each[i]=="\"":
                    splitpos.append(i)
            if len(splitpos)>=4:
                before=each[:splitpos[0]+1]
                num1=each[splitpos[0]+1:splitpos[1]]
                mid=each[splitpos[1]:splitpos[2]+1]
                num2=each[splitpos[2]+1:splitpos[3]]
                after=each[splitpos[3]:]
                num1=indexDict[num1]
                num2=indexDict[num2]
                if num1==-1 or num2==-1:
                    continue
                each=before+num1+mid+num2+after
        res+=each+"\n"
    f = open(dotFiles + "/" + eachFile,"w")
    f.write(res)
    f.close()

