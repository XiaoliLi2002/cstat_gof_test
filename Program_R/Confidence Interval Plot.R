library(xlsx)
library(ggplot2)
path<-"F:/Works/Summer Research/Program_R/plotdata.xlsx" #Plot Data Path
pltdata<-read.xlsx(path,1)
ggplot(pltdata,aes(x=mean))+ #1.96 is the critical value for standard normal distribution at level 97.5%
  geom_ribbon(aes(ymin=E1-1.96*V1,ymax=E1+1.96*V1,fill="G1",color="G1"),alpha=0.3,linetype=2)+
  geom_ribbon(aes(ymin=E1-V1,ymax=E1+(V1),fill="G1",color="G1"),alpha=0.7,linetype=2)+
  geom_ribbon(aes(ymin=E2-1.96*(V2),ymax=E2+1.96*(V2),fill="G2",color="G2"),alpha=0.3,linetype=2)+
  geom_ribbon(aes(ymin=E2-(V2),ymax=E2+(V2),fill="G2",color="G2"),alpha=0.7,linetype=2)+
  geom_ribbon(aes(ymin=73.36108019128368,ymax=128.4219886438403,fill="G3",color="G3"),alpha=0.3,linetype=2)+ 
  geom_ribbon(aes(ymin=85.04690278520079,ymax=112.93897468761988,fill="G3",color="G3"),alpha=0.7,linetype=2)+
  geom_line(aes(y=E1,color="G1"))+ #Theory
  geom_line(aes(y=E2,color="G2"))+ #Bootstrap
  geom_line(aes(y=99,color='G3'))+ #G3 is 95% confidence interval of \chi^2 distribution with df=99
  theme_bw(base_family = "Times")+
  scale_fill_discrete(labels=c("Theoretical","Bootstrap", expression(~~chi^2~(n-1))))+
  scale_color_discrete(labels=c("Theoretical","Bootstrap", expression(~~chi^2~(n-1))))+
  theme(panel.grid = element_blank(),
        legend.position = "top",                      # legend top
        panel.border = element_blank(),
        text = element_text(family = "STHeiti"),      
        plot.title = element_text(hjust = 0.5)) +     
  labs(x = 'mean', y = expression(C[min]), title = ("Confidence Intervals(n=100) of constant model"),
       color = "", fill = "")
  