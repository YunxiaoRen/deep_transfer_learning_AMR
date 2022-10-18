library(ggpubr)
library(ggplot2)
library(ggsci)
library(plyr)


gi_data <- read.table("gi_data_all_metrics.txt",header = T)
gi_data

compare_means(MCC ~ Model, gi_data, method = "t.test", group.by = "Drug")
compare_means(F1_R ~ Model, gi_data, method = "t.test", group.by = "Drug")

### bar plot
p1 <- ggbarplot(gi_data,x="Drug",y="MCC",color = "Model",palette = "jco", add = "mean_se",
          position = position_dodge(0.8))+
          stat_compare_means(aes(group = Model), method = "t.test",label = "p.signif",label.y =0.65)+
          scale_y_continuous(expand=c(0.02,0),limits=c(0, 0.7),breaks=seq(0,1,0.1))


p2 <- ggbarplot(gi_data,x="Drug",y="F1_R",color = "Model",palette = "jco", add = "mean_se",
                position = position_dodge(0.8))+
  stat_compare_means(aes(group = Model), method = "t.test",label = "p.signif",label.y =0.85)+
  scale_y_continuous(expand=c(0.02,0),limits=c(0, 1),breaks=seq(0,1,0.1))


library(patchwork)
p1 + p2 + plot_layout(nrow=1,guides = 'collect')
ggsave("Fig5_MCC_F1.pdf",width = 8, height = 3.5,dpi = 300)

