
basic_history <- read.csv("GEN_basic_model_history_out.csv")

temp= data.frame()
for (i in seq(5)){
  k_loss_1 = basic_history[i,1]
  k_loss_2 = gsub("\\[","",k_loss_1)
  k_loss_2 = gsub("\\]","",k_loss_2)
  k_loss_3 = as.numeric(unlist(strsplit(k_loss_2,", ")))
  k_loss_df = data.frame(loss=k_loss_3,K_fold=i)
  loss = rbind(k_loss_df,temp)
  temp = loss
}

temp= data.frame()
for (i in seq(5)){
  k_acc_1 = basic_history[i,2]
  k_acc_2 = gsub("\\[","",k_acc_1)
  k_acc_2 = gsub("\\]","",k_acc_2)
  k_acc_3 = as.numeric(unlist(strsplit(k_acc_2,", ")))
  k_acc_df = data.frame(acc=k_acc_3,K_fold=i)
  acc = rbind(k_acc_df,temp)
  temp = acc
}
temp= data.frame()	
for (i in seq(5)){
  k_f1_m_1 = basic_history[i,3]
  k_f1_m_2 = gsub("\\[","",k_f1_m_1)
  k_f1_m_2 = gsub("\\]","",k_f1_m_2)
  k_f1_m_3 = as.numeric(unlist(strsplit(k_f1_m_2,", ")))
  k_f1_m_df = data.frame(f1_m=k_f1_m_3,K_fold=i)
  f1_m = rbind(k_f1_m_df,temp)
  temp = f1_m
}
temp= data.frame()
for (i in seq(5)){
  k_precision_m_1 = basic_history[i,4]
  k_precision_m_2 = gsub("\\[","",k_precision_m_1)
  k_precision_m_2 = gsub("\\]","",k_precision_m_2)
  k_precision_m_3 = as.numeric(unlist(strsplit(k_precision_m_2,", ")))
  k_precision_m_df = data.frame(precision_m=k_precision_m_3,K_fold=i)
  precision_m = rbind(k_precision_m_df,temp)
  temp = precision_m
}
temp= data.frame()
for (i in seq(5)){
  k_recall_m_1 = basic_history[i,5]
  k_recall_m_2 = gsub("\\[","",k_recall_m_1)
  k_recall_m_2 = gsub("\\]","",k_recall_m_2)
  k_recall_m_3 = as.numeric(unlist(strsplit(k_recall_m_2,", ")))
  k_recall_m_df = data.frame(recall_m=k_recall_m_3,K_fold=i)
  recall_m = rbind(k_recall_m_df,temp)
  temp = recall_m
}

temp= data.frame()	
for (i in seq(5)){
  k_val_loss_1 = basic_history[i,6]
  k_val_loss_2 = gsub("\\[","",k_val_loss_1)
  k_val_loss_2 = gsub("\\]","",k_val_loss_2)
  k_val_loss_3 = as.numeric(unlist(strsplit(k_val_loss_2,", ")))
  k_val_loss_df = data.frame(val_loss=k_val_loss_3,K_fold=i)
  val_loss = rbind(k_val_loss_df,temp)
  temp = val_loss
}

temp= data.frame()
for (i in seq(5)){
  k_val_acc_1 = basic_history[i,7]
  k_val_acc_2 = gsub("\\[","",k_val_acc_1)
  k_val_acc_2 = gsub("\\]","",k_val_acc_2)
  k_val_acc_3 = as.numeric(unlist(strsplit(k_val_acc_2,", ")))
  k_val_acc_df = data.frame(val_acc=k_val_acc_3,K_fold=i)
  val_acc = rbind(k_val_acc_df,temp)
  temp = val_acc
}

temp= data.frame()
for (i in seq(5)){
  k_val_f1_m_1 = basic_history[i,8]
  k_val_f1_m_2 = gsub("\\[","",k_val_f1_m_1)
  k_val_f1_m_2 = gsub("\\]","",k_val_f1_m_2)
  k_val_f1_m_3 = as.numeric(unlist(strsplit(k_val_f1_m_2,", ")))
  k_val_f1_m_df = data.frame(val_f1_m=k_val_f1_m_3,K_fold=i)
  val_f1_m = rbind(k_val_f1_m_df,temp)
  temp = val_f1_m
}

temp= data.frame()
for (i in seq(5)){
  k_val_precision_m_1 = basic_history[i,9]
  k_val_precision_m_2 = gsub("\\[","",k_val_precision_m_1)
  k_val_precision_m_2 = gsub("\\]","",k_val_precision_m_2)
  k_val_precision_m_3 = as.numeric(unlist(strsplit(k_val_precision_m_2,", ")))
  k_val_precision_m_df = data.frame(val_precision_m=k_val_precision_m_3,K_fold=i)
  val_precision_m = rbind(k_val_precision_m_df,temp)
  temp = val_precision_m
}

temp= data.frame()
for (i in seq(5)){
  k_val_recall_m_1 = basic_history[i,10]
  k_val_recall_m_2 = gsub("\\[","",k_val_recall_m_1)
  k_val_recall_m_2 = gsub("\\]","",k_val_recall_m_2)
  k_val_recall_m_3 = as.numeric(unlist(strsplit(k_val_recall_m_2,", ")))
  k_val_recall_m_df = data.frame(val_recall_m=k_val_recall_m_3,K_fold=i)
  val_recall_m = rbind(k_val_recall_m_df,temp)
  temp = val_recall_m
}	

metrics = cbind(loss,val_loss,acc,val_acc,f1_m,val_f1_m,precision_m,val_precision_m,recall_m,val_recall_m)
head(metrics)
metrics2 = metrics[,-c(2,4,6,8,10,12,14,16,18)]
head(metrics2)
dim(metrics2)

write.table(metrics2,file="GEN_basic_CNN_metrics.txt",row.names=F,sep='\t',quote=F)


  

