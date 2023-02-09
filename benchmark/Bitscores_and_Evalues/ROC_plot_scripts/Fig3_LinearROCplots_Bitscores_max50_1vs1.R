
##########---------------------------Linear bitscores TPs_FPs-plots--------------

#R CMD BATCH Fig3_LinearROCplots_Bitscores_max50_1vs1.R

##########-----INPUT and OUTPUT files-------------

Input_dir="../ROC_plot_Bitscores_max50_1vs1_rank/"
Output_dir="../plots_example/"

##########------------------
datb = c("_pfam_max50_1vs1","_supfam_max50_1vs1","_gene3d_max50_1vs1")
datb2 = c("A) PFAM + Clan","B) SCOP/Superfamily","C) CATH/Gene3D")

meth1 = c("csblast","phmmer","hhsearch","blast","usearch","fasta","ublast")
meth2 = c("phmmer","csblast","hhsearch","blast","usearch","fasta","ublast")
meth3 = c("csblast","phmmer","hhsearch","blast","usearch","fasta","ublast")

lntp11 = c(1,1,1,1,1,1,1,1)
lntp12 = c(1,1,1,1,1,1,1,1)
lntp13 = c(1,1,1,1,1,1,1,1)
lntp2 = c(3,3,3,3,3,3,3,1)

plot_colors1 <-  c("#D95F0E","#EEAD0E","#4DAC26","#92C5DE","#636363","#D01C8B","#0eeead","#0eeead")
plot_colors2 <-  c("#EEAD0E","#D95F0E","#4DAC26","#92C5DE","#636363","#D01C8B","#0eeead","#0eeead")
plot_colors3 <-  c("#D95F0E","#EEAD0E","#4DAC26","#92C5DE","#636363","#D01C8B","#0eeead","#0eeead")

##########---Insight----bitscores, AUC values for the first 1000 FPs------------

lnames_pf <- c('CSBLAST:     0.9236','PHMMER:     0.9226','HHSEARCH: 0.9167','NCBI-BLAST:0.9087','USEARCH:    0.8956','FASTA:           0.8879','UBLAST:        0.8561')
lnames_sf <- c('PHMMER:     0.9031','CSBLAST:     0.8999','HHSEARCH: 0.8774','NCBI-BLAST:0.8570','USEARCH:    0.8533','FASTA:           0.8343','UBLAST:        0.7846')
lnames_gd <- c('CSBLAST:     0.9100','PHMMER:     0.9032','HHSEARCH: 0.9009','NCBI-BLAST:0.8784','USEARCH:    0.8613','FASTA:           0.8523','UBLAST:        0.8158')


##########---NOTE: Here, you could insert axis range of TPs and FPs. Ex: "xlim = c(1,1000)"

######-----subplots--------------

subx <- function() for (j in 1:3) {
  for(i in 1:length(meth1)){
    oname = paste(Input_dir, meth1[i], sep="")
    cumdata_log <- assign(oname, read.table(paste(oname, datb[j], sep="")))
    sens <- cumdata_log[,1]
    spec <- cumdata_log[,2]
    if ( j == 1){
      if (meth1[i]=="csblast"){
        plot(spec,sens, type="l",lty=lntp11[i],lwd=lntp2[i], col=plot_colors1[i],main = datb2[j],xlab= "False positives",ylab= "True positives",ylim = c(2944,5244))#,xlim = c(0,1000)) 
      } else {
        lines(spec,sens, type="l",lty=lntp11[i],lwd=lntp2[i], col=plot_colors1[i])
      }
    }
}
}

suby <- function() for (j in 1:3) {
  for(i in 1:length(meth2)){
    oname = paste(Input_dir, meth2[i], sep="")
    cumdata_log <- assign(oname, read.table(paste(oname, datb[j], sep="")))
    sens <- cumdata_log[,1]
    spec <- cumdata_log[,2]
    if ( j == 2){
      if (meth2[i]=="phmmer"){
        plot(spec,sens, type="l",lty=lntp12[i],lwd=lntp2[i], col=plot_colors2[i],main = datb2[j],xlab= "False positives",ylab= "True positives",ylim = c(2246,5046))#,xlim = c(0,1000))  
      } else {
        lines(spec,sens, type="l",lty=lntp12[i],lwd=lntp2[i], col=plot_colors2[i])
      }
    }
  }
}

subz <- function() for (j in 1:3) {
  for(i in 1:length(meth3)){
    oname = paste(Input_dir, meth3[i], sep="")
    cumdata_log <- assign(oname, read.table(paste(oname, datb[j], sep="")))
    sens <- cumdata_log[,1]
    spec <- cumdata_log[,2]
    if ( j == 3){
      if (meth3[i]=="csblast"){
        plot(spec,sens, type="l",lty=lntp13[i],lwd=lntp2[i], col=plot_colors3[i],main = datb2[j],xlab= "False positives",ylab= "True positives",ylim = c(3255,5655))#,xlim = c(0,1000)) 
      } else {
        lines(spec,sens, type="l",lty=lntp13[i],lwd=lntp2[i], col=plot_colors3[i])
      }
    }
  }
}

pdf(paste(Output_dir,"Linear_bitscores_max50_1vs1_pfam.pdf", sep = ""))
subx()
legend('bottomright', lnames_pf,title = "Search tools: AUC1000 scores", lty = lntp11,col = plot_colors1,cex=1.1,inset= 0.00,lwd = 3)
dev.off()

pdf(paste(Output_dir,"Linear_bitscores_max50_1vs1_supfam.pdf", sep = "")) 
suby()	
legend('bottomright', lnames_sf,title = "Search tools: AUC1000 scores", lty = lntp12,col = plot_colors2,cex=1.1,inset= 0.00,lwd = 3)
dev.off()

pdf(paste(Output_dir,"Linear_bitscores_max50_1vs1_gene3d.pdf", sep = ""))
subz()
legend('bottomright', lnames_gd,title = "Search tools: AUC1000 scores", lty = lntp13,col = plot_colors3,cex=1.1,inset= 0.00,lwd = 3)
dev.off()




