# create new features in train and test set

feats <- c('f274','f275','f276','f277')
for (set in c('train','test')) {
    for (feat in feats) {
        cat("Started calculating new features from ordering by features ",feat,", f521 in ",set," dataset.\n",sep="")

        data <- read.csv( paste("./data/",set,"_v2.csv",sep=""), sep=",", header=T )
        data <- data[,c('id','f67','f68','f224','f256','f274','f275','f276','f277','f521','f527','f528','f529','f530','f539','f585')]  

        data[,8] <- as.numeric(as.character(data[,8]))
        data[,9] <- as.numeric(as.character(data[,9]))
        data[,16] <- as.numeric(as.character(data[,16]))
        
        data <- lapply(data, function(x) {mn <- median(x,na.rm=T);x[is.na(x)] <- mn;x})
        data <- as.data.frame(data)
        library(data.table)
        data <- data.table(data, key = c(feat, 'f521'))
        data <- as.data.frame(data)
        datalength <- dim(data)[1]

        # bind ids and columns with zeroes to create dateframe
        df <- cbind(data$id, rep(0,datalength), rep(0,datalength), rep(0,datalength), rep(0,datalength), rep(0,datalength), rep(0,datalength))
        df <- as.data.frame(df)

        # remove noise
        te1 <- data$f527/data$f529
        te1[is.na(te1) | is.infinite(te1)] <- median(te1, na.rm=T)
        te2 <- data$f528/data$f529
        te2[is.na(te2) | is.infinite(te2)] <- median(te2, na.rm=T)
        te3 <- data$f528/data$f527
        te3[is.na(te3) | is.infinite(te3)] <- median(te3, na.rm=T)
        te4 <- data$f530/data$f527
        te4[is.na(te4) | is.infinite(te4)] <- 0
        te5 <- data$f67/(data$f68+1)
        te5[is.na(te5) | is.infinite(te5)] <- median(te5, na.rm=T)
        teindex <- seq(1,datalength)
        tele <- datalength

        nudata <- data[
          (((data$f585 < 2e+16)  & (teindex <= 0.5*tele)) | (teindex > 0.5*tele)) & 
            (((data$f528-data$f527 < 100000)  & (teindex <= 0.5*tele)) | (teindex > 0.5*tele)) & 
            ((teindex <= 0.15*tele) | ((teindex > 0.15*tele) & (te3 < 1.4))) & 
            ((teindex <= 0.15*tele) | ((teindex > 0.15*tele) & (te2-te1 < 1.5)) ) & 
            ((teindex <= 0.15*tele) | ((teindex > 0.15*tele) & (te1 < 8)) ) & 
            ((teindex <= 0.15*tele) | ((teindex > 0.15*tele) & (te1 > 2)) ) & 
            (te5 < 0.9) & 
            (te1 >= 1) & 
            (te1 < 25) & 
            (data$f528-data$f527 >= 0) &
            (data$f528-data$f527 < 400000) &
            ((te2-te1) >= 0) & 
            ((te2-te1) < 3) & 
            (te3 >= 1) &
            (te3 <= 1.8) & 
            (te4 < 6000), ]
        cat(dim(nudata))
        nudatalength <- dim(nudata)[1]

        # get the data
        diffdata <- ((nudata[1:(nudatalength-1), 2:16] < nudata[2:nudatalength, 2:16]) | (nudata[1:(nudatalength-1), 2:16] > nudata[2:nudatalength, 2:16])) + 0
        diffdata <- as.data.frame(diffdata)
        f224last <- c(diffdata$f224,0)
        f256last <- c(diffdata$f256,0)
        f539last <- c(diffdata$f539,0)
        # TODO : if previous f224 is different from this one, then set it to 0
        diffdata <- nudata[2:nudatalength, 2:16] - nudata[1:(nudatalength-1), 2:16]
        diffdata <- as.data.frame(diffdata)
        f527sub <- c(0, diffdata$f527)
        f528sub <- c(0, diffdata$f528)

        # insert by id
        ind <- sapply(nudata$id, function(x) {which(df[,1] == x)})
        df[ind, 2] <- f224last
        df[ind, 3] <- f256last
        df[ind, 4] <- f539last
        df[ind, 5] <- f527sub
        df[ind, 6] <- f528sub

        # insert normalized sequence order
        order <- seq(1, nudatalength)
        order <- order-mean(order)
        order <- order/sd(order)
        df[ind, 7] <- order

        names(df) <- c("id", "f224last", "f256last", "f539last", "f527sub", "f528sub", "seq")

        df <- data.table(df, key = c('id'))
        write.csv(df, paste("./data/",set,"_AddData3_",feat,".csv",sep=""), row.names=F)
        cat("Done.\n")
    }
}