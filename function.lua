function split(inputstr, sep)
        if sep == nil then
                sep = "%s"
        end
        local t={} ; i=1
        for str in string.gmatch(inputstr, "([^"..sep.."]+)") do
                t[i] = str
                i = i + 1
        end
        return t
end

function Generate(trainTensor,labelTrainTensor,validTensor,labelValidTensor,indxDom,Size)
        nSujet = Size
        s=1
        while (s<= nSujet) do -- s number of subject
                
filename=string.format("./data/%s%u%s%u%s","S",s,"_D",indxDom,"_specgramCH_27.csv")
                file=io.open(filename,"r") -- open file 

                frq=1
                for line in io.lines(filename) do -- line for time  
                        lineSplit=split(line,sep)
                        nLineSplit=table.getn(lineSplit)

                                for time=1,nTimeTrain  do
                                        trainTensor[s][time][frq] = lineSplit[time]
                                end

                                for time=1,nTimeValid  do
                                        validTensor[s][time][frq] = lineSplit[time+nTimeTrain]
                                end
                        frq=frq+1

                end
                labelTrainTensor[s] = s
                labelValidTensor[s] = s

                s=s+1
                io.close(file)
        end
end

function zeroMean(tensor,size,nTime)

        for n=1,tensor:size(1) do
                for t=1,tensor:size(2) do
                        s=0
                        for freq=1,tensor:size(3) do
                                s=s+tensor[n][t][freq]
                        end
                        for freq=1,tensor:size(3) do
                                tensor[n][t][freq]=tensor[n][t][freq]-(s/tensor:size(3))
                        end
                end
        end
end

function dataAugmentationSujet (trainTensor,labelTensor,tensor,nbSujet,posIn,nTime)
        pos = posIn
        time=1
        for t=1,nTime do
                for frq=1, nFreq do
                        trainTensor[pos][time][frq] = tensor[nbSujet][t][frq]
                end
                time = time + 1
                if time > ntime then
                        time =1
                        labelTensor[pos]=nbSujet
                        pos =pos + 1
                end
        end
end

function dataAugmentationSujet2 (resTrainTensor,resLabelTensor,tensor,tensorLabel,posIn,pos,nTime,ntime)
        i = posIn
        j = pos
        bord = posIn+ nTime/ntime -1
        while (i<=bord) do
                if i == posIn + nTime/ntime then

                        for t=1,ntime-1 do
                                resTrainTensor[j][t]:copy(tensor[i][t+1])
				print('ok')
                        end

                        resTrainTensor[j][t+1]:copy(tensor[posIn][1])
                else

                        for t=1, ntime do
                                if t == ntime then
                                        resTrainTensor[j][t]:copy(tensor[i+1][1])
                                else
                                        resTrainTensor[j][t]:copy(tensor[i][t+1])
                                end
                        end

                end

                resLabelTensor[j]=tensorLabel[posIn]
                i = i + 1
                j = j + 1
        end
end

function alea(tensor, tensorLabel,tensor1, tensorLabel1, Size)
        shuffle = torch.randperm(Size)

        aux=torch.CudaTensor(1,tensor:size(2),tensor:size(3))
        aux1=torch.CudaTensor(1,tensor:size(2),tensor:size(3))

        for i=1, Size do
                aux:copy(tensor[shuffle[i]])
                tensor[shuffle[i]]:copy(tensor[i])
                tensor[i]:copy(aux)
                auxLabel=tensorLabel[i]
                tensorLabel[i]=tensorLabel[shuffle[i]]
                tensorLabel[shuffle[i]]=auxLabel

                aux1:copy(tensor1[shuffle[i]])
                tensor1[shuffle[i]]:copy(tensor1[i])
                tensor1[i]:copy(aux1)
                auxLabel1=tensorLabel1[i]
                tensorLabel1[i]=tensorLabel1[shuffle[i]]
                tensorLabel1[shuffle[i]]=auxLabel1
        end


end

sep = ','

function noiseAdd(tensor,indice,pos)
        noise=torch.CudaTensor(ntime,nFreq)

        noise:normal(0,0.001)
        for t=1,ntime do
                for frq=1,nFreq do
                        tensor[pos][t][frq]=tensor[indice][t][frq]+noise[t][frq]
                end
        end
end

function generateNoise(tensorTrain,tensorLabel)

	posIn =829
	ind =1
        for pos= posIn, posIn + 450 -1  do
                noiseAdd(tensorTrain,ind,pos)
		tensorLabel[pos]=tensorLabel[ind]
		ind = ind + 1 
        end
end

function maxObs(tensor,maxTensor)
	for i=1, tensor:size(1) do 
		for j=1, tensor:size(2) do 
			mAX = torch.max(tensor[i][j])
			mIN =torch.min(tensor[i][j])
			if (mAX >= torch.abs(mIN)) then
				maxTensor[i][j][1] = mAX
			else 
				maxTensor[i][j][1] = torch.abs(mIN)
			end
		end
	end
end


function rescal(tensor,maxTensor) 
	for i=1,tensor:size(1) do 
		for j=1,tensor:size(2) do 
			for k=1,tensor:size(3) do 
				tensor[i][j][k] = tensor[i][j][k]/maxTensor[i][j][1]
			end
		end
	end
end

function Rescal(sourceTensor,validTensor,moyenneTensor,stdTensor)
	for f=1,sourceTensor:size(3) do	
		sujet=1		
		for i=1,sourceTensor:size(1) do
			for j=1,sourceTensor:size(2) do
				moyenneTensor[sujet][f]=moyenneTensor[sujet][f]+sourceTensor[i][j][f]
			end
			if (i%(nTimeTrain/ntime) == 0) then sujet = sujet + 1 end
		end

		sujet1=1
		for i=1,validTensor:size(1) do
			for j=1,validTensor:size(2) do 
				moyenneTensor[sujet1][f] = moyenneTensor[sujet1][f] + validTensor[i][j][f]
			end
			if (i%(nTimeValid/ntime) == 0) then sujet1=sujet1 + 1 end

		end
	end
	for s=1,9 do
	for k=1,moyenneTensor:size(1) do 
		moyenneTensor[s][k]=moyenneTensor[s][k]/580
	end
	end

	for f=1,sourceTensor:size(3) do 
		sujet=1
		for i=1,sourceTensor:size(1) do

			for j=1,sourceTensor:size(2) do 
				stdTensor[sujet][f]=stdTensor[sujet][f]+(sourceTensor[i][j][f]-moyenneTensor[sujet][f])*(sourceTensor[i][j][f]-moyenneTensor[sujet][f])
			end
			if (i%(nTimeTrain/ntime)==0) then sujet = sujet +1 end
		end
		sujet1=1
		for i=1,validTensor:size(1) do 
			for j=1,validTensor:size(2) do 
				stdTensor[sujet1][f] = stdTensor[sujet1][f]+(validTensor[i][j][f]-moyenneTensor[sujet1][f])*(validTensor[i][j][f]-moyenneTensor[sujet1][f])
			end
			if (i%(nTimeValid/ntime)==0) then sujet1 = sujet1 + 1 end 
		end
	end
print('ok')
for s=1,9 do
	for k=1,stdTensor:size(1) do 
		stdTensor[s][k] = stdTensor[s][k]/(580-1)
	end
end

for s=1,9 do 
	for k=1,stdTensor:size(1) do 
		stdTensor[s][k]=torch.sqrt(stdTensor[s][k])
	end
end

	for f=1,sourceTensor:size(3) do 
		sujet =1
		for i=1,sourceTensor:size(1) do
			for j=1,sourceTensor:size(2) do 
				sourceTensor[i][j][f]=(sourceTensor[i][j][f]-moyenneTensor[sujet][f])/stdTensor[sujet][f]
			end
			if (i%(nTimeTrain/ntime) == 0) then sujet = sujet + 1 end 
		end

	end


	for f=1,validTensor:size(3) do
		sujet1=1 
		for i=1,validTensor:size(1) do
			for j=1,validTensor:size(2) do 
				validTensor[i][j][f]=(validTensor[i][j][f]-moyenneTensor[sujet1][f])/stdTensor[sujet1][f]
			end
			if (i%(nTimeValid/ntime) == 0) then sujet1 = sujet1 +1 end
		end
	end

end
