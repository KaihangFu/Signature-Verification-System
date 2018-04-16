
%this script is for writer-dependent SV experiment

for th=1:10
    
	cth=int2str(th);
	
	%for trainpart
    cd(['F:\ISU course\thesis\signature\database\trainingSet2011\OfflineSignatures\Dutch\New NN data\user',cth,'\experimentM\trainpart']);
    files=dir('*.PNG');
    L=size(files,1);
    data=zeros(10,57);
    num=1;
    for i=1:L
        if(size(files(i).name,2)==10)
	        for j=i+1:L
	            data(num,1:56) = abs(GetFeature(files(i).name) - GetFeature(files(j).name));
			    if(size(files(j).name,2)==10)
			        data(num,57) = 1;
			    else
			        data(num,57) = 0;
			    end
			    num=num+1;
	        end
	    end
    end
    csvwrite(['F:\ISU course\thesis\signature\lab\labuser',cth,'\user',cth,'traindataM.csv'],data);

    %for testpart geniue
    cd(['F:\ISU course\thesis\signature\database\trainingSet2011\OfflineSignatures\Dutch\New NN data\user',cth,'\experimentM\testpart\geniue']);
    files=dir('*.PNG');
    L=size(files,1);
    dataG=zeros(10,57);
    num=1;
    for i=1:2
        if(size(files(i).name,2)==11)
	        for j=3:L
	            dataG(num,1:56) = abs(GetFeature(files(i).name) - GetFeature(files(j).name));
			    dataG(num,57) = 1;
			    num=num+1;
	        end
	    end
    end

    %for testpart forgeries
    cd(['F:\ISU course\thesis\signature\database\trainingSet2011\OfflineSignatures\Dutch\New NN data\user',cth,'\experimentM\testpart\forgereis']);
    files=dir('*.PNG');
    L=size(files,1);
    dataF=zeros(10,57);
    num=1;
    for i=1:4
        if(files(i).name(1,2)=='00')
	        for j=5:L
	            dataF(num,1:56) = abs(GetFeature(files(i).name) - GetFeature(files(j).name));
			    dataF(num,57) = 0;
			    num=num+1;
	        end
	    end
    end
    
	dataT = [dataF;dataG];
	csvwrite(['F:\ISU course\thesis\signature\lab\labuser',cth,'\user',cth,'testdataM.csv'],dataT);

end