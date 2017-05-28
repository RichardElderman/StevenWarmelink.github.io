/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package featuregenerator;

import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.Toolkit;
import java.awt.datatransfer.Clipboard;
import java.awt.datatransfer.StringSelection;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import javax.swing.UIManager;
import javax.swing.*;
import java.util.*;

/**
 *
 * @author Hirad Gorgoroth
 */
public class mainFrame extends javax.swing.JFrame {

    private ArrayList<JTextField> nods;
    private int inputWidth;
    private int inputHeight;
    private int thickness;
    private int [][] dimentions;
    
    
    public mainFrame() {
        frameSetup();
        generateNods(10 , 10);
        
    }
    private void generateNods(int height, int weidth){
        resultPanel.setLayout(new GridBagLayout());
        GridBagConstraints c = new GridBagConstraints();
        
        
        for(int i=0;i<=height-1;i++){
            for(int j=0; j<=weidth-1;j++){
              c.fill=GridBagConstraints.HORIZONTAL;
              c.gridx =j;
              c.gridy=i;
              JTextField jtextField = new JTextField();
              jtextField.setColumns(2);
              jtextField.setText("-1");
              nods.add(jtextField);
              jtextField.setHorizontalAlignment(JTextField.CENTER);
              resultPanel.add(nods.get(nods.size()-1),c);
            }
        }
//          JButton button = new JButton("See the Code");
//        button.addActionListener(new ActionListener() {
//            @Override
//            public void actionPerformed(ActionEvent ae) {
//            
//            }
//        });
//         c.fill = GridBagConstraints.HORIZONTAL;
//         c.gridx=0;
//         c.gridy=height+6;
//         resultPanel.add(button, c);
        
        this.pack();
        

    }
    
    private void setPattern(){
        if(manualb.isSelected()){
            manualb.setSelected(false);
            for(int i=0;i<=nods.size()-1;i++){
                nods.get(i).setText("-1");
            }
        }else{
            automatic.setSelected(false);
            dimentions = new int[inputHeight][inputWidth];
            switch (mapCombo.getSelectedIndex())
            {

                case 0: //Vertical - Middle 
                {
                    System.out.println(mapCombo.getSelectedItem());
                    
                    
                   
                   for(int i=inputHeight/2-1; i<=inputHeight/2+thickness-1;i++){
                   for(int j=0;j<=inputWidth-1;j++){
                       dimentions[i][j]=1;
                   }     
                    }
                    for(int i=0; i<=inputHeight/2-1;i++){
                   for(int j=0;j<=inputWidth-1;j++){
                       dimentions[i][j]=-1;
                   }     
                    } for(int i=inputHeight/2+thickness; i<=inputHeight-1;i++){
                   for(int j=0;j<=inputWidth-1;j++){
                       dimentions[i][j]=-1;
                   }     
                    }
                   
                     
                    int counter=0;
                    for(int i=0; i<=inputHeight-1;i++){
                        System.out.println();
                   for(int j=0;j<=inputWidth-1;j++){
                       System.out.print(dimentions[i][j]);
                       nods.get(counter).setText(""+dimentions[i][j]);
                       counter++;
                   }     
                    } 
                    break;
                }
                case 1: //Horizontal - Middle
                {
                    System.out.println(mapCombo.getSelectedItem());
                    
                    for(int i=0; i<=inputHeight-1;i++){
                   for(int j=inputWidth/2-1;j<=inputWidth/2+thickness-1;j++){
                       dimentions[i][j]=1;
                   }     
                    }
                    for(int i=0; i<=inputHeight-1;i++){
                   for(int j=0;j<=inputWidth/2-1;j++){
                       dimentions[i][j]=-1;
                   }     
                    } for(int i=0; i<=inputHeight-1;i++){
                   for(int j=inputWidth/2+thickness;j<=inputWidth-1;j++){
                       dimentions[i][j]=-1;
                   }     
                    }
                   
                     
                    int counter=0;
                    for(int i=0; i<=inputHeight-1;i++){
                        System.out.println();
                   for(int j=0;j<=inputWidth-1;j++){
                       System.out.print(dimentions[i][j]);
                       nods.get(counter).setText(""+dimentions[i][j]);
                       counter++;
                   }     
                    } 
                    break;
                }
                case 2: //Right Edge 
                {System.out.println(mapCombo.getSelectedItem());
                    for(int i=0; i<=inputHeight-1;i++){
                   for(int j=0;j<=thickness-1;j++){
                       dimentions[i][j]=1;
                   }     
                    }
                    for(int i=0; i<=inputHeight-1;i++){
                   for(int j=thickness;j<=inputWidth-1;j++){
                       dimentions[i][j]=-1;
                   }     
                    } 
                    
                    int counter=0;
                    for(int i=0; i<=inputHeight-1;i++){
                        System.out.println();
                   for(int j=0;j<=inputWidth-1;j++){
                       System.out.print(dimentions[i][j]);
                       nods.get(counter).setText(""+dimentions[i][j]);
                       counter++;
                   }     
                    } 
                    
                    break; 
                }
                case 3: //Left Edge
                {
                    
                    System.out.println(mapCombo.getSelectedItem());
                    for(int i=0; i<=inputHeight-1;i++){
                   for(int j=inputWidth-1;j>=inputWidth-thickness-1;j--){
                       dimentions[i][j]=1;
                   }     
                    }
                    for(int i=0; i<=inputHeight-1;i++){
                   for(int j=0;j<=inputWidth-thickness-1;j++){
                       dimentions[i][j]=-1;
                   }     
                    } 
                    
                    int counter=0;
                    for(int i=0; i<=inputHeight-1;i++){
                        System.out.println();
                   for(int j=0;j<=inputWidth-1;j++){
                       System.out.print(dimentions[i][j]);
                       nods.get(counter).setText(""+dimentions[i][j]);
                       counter++;
                   }     
                    } 
                    
                    break;
                }
               case 4: //Top Edge
                {
                    System.out.println(mapCombo.getSelectedItem());
                    for(int i=0; i<=thickness-1;i++){
                   for(int j=0;j<=inputWidth-1;j++){
                       dimentions[i][j]=1;
                   }     
                    }
                    for(int i=thickness; i<=inputHeight-1;i++){
                   for(int j=0;j<=inputWidth-1;j++){
                       dimentions[i][j]=-1;
                   }     
                    } 
                    
                    int counter=0;
                    for(int i=0; i<=inputHeight-1;i++){
                        System.out.println();
                   for(int j=0;j<=inputWidth-1;j++){
                       System.out.print(dimentions[i][j]);
                       nods.get(counter).setText(""+dimentions[i][j]);
                       counter++;
                   }     
                    } 
                    
                    break;
                }
                case 5: //Bottom Edge
                {
                    System.out.println(mapCombo.getSelectedItem());
                    for(int i=inputHeight-1; i>=inputHeight-thickness-1;i--){
                   for(int j=0;j<=inputWidth-1;j++){
                       dimentions[i][j]=-1;
                   }     
                    }
                    for(int i=0; i<=inputHeight-thickness-1;i++){
                   for(int j=0;j<=inputWidth-1;j++){
                       dimentions[i][j]=1;
                   }     
                    } 
                    
                    int counter=0;
                    for(int i=0; i<=inputHeight-1;i++){
                        System.out.println();
                   for(int j=0;j<=inputWidth-1;j++){
                       System.out.print(dimentions[i][j]);
                       nods.get(counter).setText(""+dimentions[i][j]);
                       counter++;
                   }     
                    } 
                    break;
                }
                case 6: //" \ "
                {
                    System.out.println(mapCombo.getSelectedItem());
                    int temp=0;
                    for(int i=0; i<=inputHeight-1;i++){
                   for(int j=0;j<=temp;j++){
                       dimentions[i][j]=-1;
                   }
                   for(int j=temp;j<=thickness;j++){
                       dimentions[i][j]=1;
                       
                   }
                   temp++;
                    }
                     
                    
                    int counter=0;
                    for(int i=0; i<=inputHeight-1;i++){
                        System.out.println();
                   for(int j=0;j<=inputWidth-1;j++){
                       System.out.print(dimentions[i][j]);
                       nods.get(counter).setText(""+dimentions[i][j]);
                       counter++;
                   }     
                    } 
                    break;
                }
                case 7: //" / "
                {
                    System.out.println(mapCombo.getSelectedItem());
                    
                    break;
                }
            }
            
        }
    }
    
    
    private void generateCode(){
        int [][]tempDimen =readInput();
         
        String code="featureMap = np.matrix([\n";
        for(int i=0;i<=inputHeight-1;i++){
            
            code=code+"                       [";
            for(int j=0;j<=inputWidth-2;j++){
                code=code+tempDimen[i][j]+",";
            }
            code=code+tempDimen[i][inputWidth-1]+"]";
            if(i !=inputHeight-1){
                code=code+",\n";
            }
        }
        
        code=code+"])";
        
        
        System.out.println(code);
        codeTA.setText(code);
    }
    private int [][]readInput(){
        
        
        int [][]tempDimen =new int[inputHeight][inputWidth];
        
        int counter=0;
        
        for(int colCouner=0;colCouner<=inputHeight-1;colCouner++){
           for(int rowCounter=0;rowCounter<=inputWidth-1;rowCounter++){ 
       
           
            tempDimen[colCouner][rowCounter]=Integer.parseInt(nods.get(counter).getText());
            counter++;
               System.out.println(tempDimen[colCouner][rowCounter]);
        } 
        }
       for(int i=0;i<=inputHeight-1;i++){
           for(int j=0;j<=inputWidth-1;j++){
                System.out.println(tempDimen[i][j]);
           }
       }
        return tempDimen;
    }
    private void removeNodes(){
        for(int i=0;i<=nods.size()-1;i++){
            resultPanel.remove(nods.get(i));
        }
        resultPanel.revalidate();
        resultPanel.repaint();
        nods.retainAll(nods);
        nods=new ArrayList<>();
        System.out.println(nods.size());
    }
    
    /**
     * This method is called from within the constructor to initialize the form.
     * WARNING: Do NOT modify this code. The content of this method is always
     * regenerated by the Form Editor.
     */
    @SuppressWarnings("unchecked")
    // <editor-fold defaultstate="collapsed" desc="Generated Code">//GEN-BEGIN:initComponents
    private void initComponents() {

        jPanel1 = new javax.swing.JPanel();
        featurePanel = new javax.swing.JPanel();
        resultPanel = new javax.swing.JPanel();
        jPanel6 = new javax.swing.JPanel();
        jButton1 = new javax.swing.JButton();
        jButton2 = new javax.swing.JButton();
        widthTa = new javax.swing.JTextField();
        HieghtTa = new javax.swing.JTextField();
        jLabel1 = new javax.swing.JLabel();
        jLabel2 = new javax.swing.JLabel();
        manualb = new javax.swing.JToggleButton();
        jLabel3 = new javax.swing.JLabel();
        jButton3 = new javax.swing.JButton();
        mapCombo = new javax.swing.JComboBox<>();
        automatic = new javax.swing.JToggleButton();
        ThicknessTa = new javax.swing.JTextField();
        jLabel4 = new javax.swing.JLabel();
        jSeparator1 = new javax.swing.JSeparator();
        resTa = new javax.swing.JLabel();
        jPanel7 = new javax.swing.JPanel();
        jButton4 = new javax.swing.JButton();
        codePanel = new javax.swing.JPanel();
        jButton5 = new javax.swing.JButton();
        jScrollPane1 = new javax.swing.JScrollPane();
        codeTA = new javax.swing.JTextArea();
        jLabel5 = new javax.swing.JLabel();
        jButton6 = new javax.swing.JButton();
        jPanel4 = new javax.swing.JPanel();
        jPanel5 = new javax.swing.JPanel();
        jPanel2 = new javax.swing.JPanel();

        setDefaultCloseOperation(javax.swing.WindowConstants.EXIT_ON_CLOSE);

        jPanel1.setLayout(new java.awt.CardLayout());

        resultPanel.setBorder(javax.swing.BorderFactory.createBevelBorder(javax.swing.border.BevelBorder.RAISED, java.awt.Color.lightGray, java.awt.Color.lightGray, java.awt.Color.darkGray, java.awt.Color.gray));

        javax.swing.GroupLayout resultPanelLayout = new javax.swing.GroupLayout(resultPanel);
        resultPanel.setLayout(resultPanelLayout);
        resultPanelLayout.setHorizontalGroup(
            resultPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 0, Short.MAX_VALUE)
        );
        resultPanelLayout.setVerticalGroup(
            resultPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 293, Short.MAX_VALUE)
        );

        jPanel6.setBorder(javax.swing.BorderFactory.createBevelBorder(javax.swing.border.BevelBorder.RAISED));

        jButton1.setText("Generate");
        jButton1.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jButton1ActionPerformed(evt);
            }
        });

        jButton2.setText("Clear");
        jButton2.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jButton2ActionPerformed(evt);
            }
        });

        jLabel1.setText("Width:");

        jLabel2.setText("Height:");

        manualb.setText("Manual");
        manualb.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                manualbActionPerformed(evt);
            }
        });

        jLabel3.setText("Pre-defined map");

        jButton3.setText("Go to result code");
        jButton3.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jButton3ActionPerformed(evt);
            }
        });

        mapCombo.setModel(new javax.swing.DefaultComboBoxModel<>(new String[] { "Vertical - Middle ", "Horizontal - Middle", "Right Edge ", "Left Edge", "Top Edge", "Bottom Edge", "\" \\ \"", "\" / \"" }));

        automatic.setText("Automatic");
        automatic.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                automaticActionPerformed(evt);
            }
        });

        ThicknessTa.setText("*Not Required for Manual*");

        jLabel4.setText("Thickness:");

        javax.swing.GroupLayout jPanel6Layout = new javax.swing.GroupLayout(jPanel6);
        jPanel6.setLayout(jPanel6Layout);
        jPanel6Layout.setHorizontalGroup(
            jPanel6Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(jPanel6Layout.createSequentialGroup()
                .addGap(40, 40, 40)
                .addGroup(jPanel6Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.TRAILING, false)
                    .addComponent(manualb, javax.swing.GroupLayout.PREFERRED_SIZE, 221, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addGroup(jPanel6Layout.createSequentialGroup()
                        .addComponent(jLabel2)
                        .addGap(30, 30, 30)
                        .addComponent(HieghtTa, javax.swing.GroupLayout.PREFERRED_SIZE, 221, javax.swing.GroupLayout.PREFERRED_SIZE))
                    .addGroup(jPanel6Layout.createSequentialGroup()
                        .addComponent(jLabel1)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                        .addComponent(widthTa, javax.swing.GroupLayout.PREFERRED_SIZE, 221, javax.swing.GroupLayout.PREFERRED_SIZE))
                    .addGroup(jPanel6Layout.createSequentialGroup()
                        .addComponent(jLabel4)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                        .addComponent(ThicknessTa, javax.swing.GroupLayout.PREFERRED_SIZE, 221, javax.swing.GroupLayout.PREFERRED_SIZE)))
                .addGroup(jPanel6Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addGroup(jPanel6Layout.createSequentialGroup()
                        .addGap(126, 126, 126)
                        .addComponent(jLabel3)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
                    .addGroup(javax.swing.GroupLayout.Alignment.TRAILING, jPanel6Layout.createSequentialGroup()
                        .addGroup(jPanel6Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.TRAILING)
                            .addGroup(jPanel6Layout.createSequentialGroup()
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                                .addComponent(mapCombo, javax.swing.GroupLayout.PREFERRED_SIZE, 230, javax.swing.GroupLayout.PREFERRED_SIZE))
                            .addGroup(jPanel6Layout.createSequentialGroup()
                                .addGap(62, 62, 62)
                                .addComponent(automatic, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)))
                        .addGap(82, 82, 82)))
                .addGroup(jPanel6Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING, false)
                    .addGroup(jPanel6Layout.createSequentialGroup()
                        .addComponent(jButton1)
                        .addGap(31, 31, 31)
                        .addComponent(jButton2, javax.swing.GroupLayout.PREFERRED_SIZE, 84, javax.swing.GroupLayout.PREFERRED_SIZE))
                    .addComponent(jButton3, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
                .addContainerGap())
        );
        jPanel6Layout.setVerticalGroup(
            jPanel6Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(jPanel6Layout.createSequentialGroup()
                .addGap(27, 27, 27)
                .addGroup(jPanel6Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.TRAILING)
                    .addGroup(jPanel6Layout.createSequentialGroup()
                        .addGroup(jPanel6Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                            .addComponent(HieghtTa, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                            .addComponent(jLabel2))
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addGroup(jPanel6Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                            .addComponent(widthTa, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                            .addComponent(jLabel1))
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                        .addGroup(jPanel6Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                            .addComponent(ThicknessTa, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                            .addComponent(jLabel4)))
                    .addGroup(jPanel6Layout.createSequentialGroup()
                        .addComponent(jLabel3)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addComponent(mapCombo)))
                .addGap(18, 18, 18)
                .addGroup(jPanel6Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(manualb)
                    .addComponent(automatic))
                .addGap(29, 29, 29))
            .addGroup(javax.swing.GroupLayout.Alignment.TRAILING, jPanel6Layout.createSequentialGroup()
                .addContainerGap(javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                .addGroup(jPanel6Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(jButton2, javax.swing.GroupLayout.PREFERRED_SIZE, 60, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(jButton1, javax.swing.GroupLayout.PREFERRED_SIZE, 60, javax.swing.GroupLayout.PREFERRED_SIZE))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
                .addComponent(jButton3, javax.swing.GroupLayout.PREFERRED_SIZE, 41, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addContainerGap())
        );

        resTa.setText("Example\"");

        jButton4.setText("Generate Code");
        jButton4.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jButton4ActionPerformed(evt);
            }
        });

        javax.swing.GroupLayout jPanel7Layout = new javax.swing.GroupLayout(jPanel7);
        jPanel7.setLayout(jPanel7Layout);
        jPanel7Layout.setHorizontalGroup(
            jPanel7Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(jPanel7Layout.createSequentialGroup()
                .addGap(392, 392, 392)
                .addComponent(jButton4)
                .addContainerGap(409, Short.MAX_VALUE))
        );
        jPanel7Layout.setVerticalGroup(
            jPanel7Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(javax.swing.GroupLayout.Alignment.TRAILING, jPanel7Layout.createSequentialGroup()
                .addContainerGap()
                .addComponent(jButton4, javax.swing.GroupLayout.DEFAULT_SIZE, 50, Short.MAX_VALUE)
                .addContainerGap())
        );

        javax.swing.GroupLayout featurePanelLayout = new javax.swing.GroupLayout(featurePanel);
        featurePanel.setLayout(featurePanelLayout);
        featurePanelLayout.setHorizontalGroup(
            featurePanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(featurePanelLayout.createSequentialGroup()
                .addGroup(featurePanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addGroup(featurePanelLayout.createSequentialGroup()
                        .addContainerGap()
                        .addGroup(featurePanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                            .addComponent(jPanel6, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                            .addComponent(resultPanel, javax.swing.GroupLayout.Alignment.TRAILING, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)))
                    .addGroup(featurePanelLayout.createSequentialGroup()
                        .addGroup(featurePanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                            .addGroup(featurePanelLayout.createSequentialGroup()
                                .addGap(20, 20, 20)
                                .addGroup(featurePanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                                    .addGroup(featurePanelLayout.createSequentialGroup()
                                        .addGap(10, 10, 10)
                                        .addComponent(resTa))
                                    .addComponent(jSeparator1, javax.swing.GroupLayout.PREFERRED_SIZE, 868, javax.swing.GroupLayout.PREFERRED_SIZE)))
                            .addGroup(featurePanelLayout.createSequentialGroup()
                                .addContainerGap()
                                .addComponent(jPanel7, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)))
                        .addGap(0, 0, Short.MAX_VALUE)))
                .addContainerGap())
        );
        featurePanelLayout.setVerticalGroup(
            featurePanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(javax.swing.GroupLayout.Alignment.TRAILING, featurePanelLayout.createSequentialGroup()
                .addContainerGap()
                .addComponent(jPanel6, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(jSeparator1, javax.swing.GroupLayout.PREFERRED_SIZE, 10, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addGap(1, 1, 1)
                .addComponent(resTa)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
                .addComponent(resultPanel, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(jPanel7, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addContainerGap(21, Short.MAX_VALUE))
        );

        jPanel1.add(featurePanel, "card6");

        jButton5.setText("Back to Main Panel");
        jButton5.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jButton5ActionPerformed(evt);
            }
        });

        codeTA.setColumns(20);
        codeTA.setRows(5);
        jScrollPane1.setViewportView(codeTA);

        jLabel5.setText("Result Code: ");

        jButton6.setText("Copy All");
        jButton6.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jButton6ActionPerformed(evt);
            }
        });

        javax.swing.GroupLayout codePanelLayout = new javax.swing.GroupLayout(codePanel);
        codePanel.setLayout(codePanelLayout);
        codePanelLayout.setHorizontalGroup(
            codePanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(javax.swing.GroupLayout.Alignment.TRAILING, codePanelLayout.createSequentialGroup()
                .addContainerGap(50, Short.MAX_VALUE)
                .addGroup(codePanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addComponent(jLabel5)
                    .addComponent(jScrollPane1, javax.swing.GroupLayout.PREFERRED_SIZE, 854, javax.swing.GroupLayout.PREFERRED_SIZE))
                .addGap(22, 22, 22))
            .addGroup(codePanelLayout.createSequentialGroup()
                .addGap(296, 296, 296)
                .addComponent(jButton5, javax.swing.GroupLayout.PREFERRED_SIZE, 140, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addGap(104, 104, 104)
                .addComponent(jButton6, javax.swing.GroupLayout.PREFERRED_SIZE, 132, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addContainerGap(javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
        );
        codePanelLayout.setVerticalGroup(
            codePanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(codePanelLayout.createSequentialGroup()
                .addContainerGap(70, Short.MAX_VALUE)
                .addComponent(jLabel5)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(jScrollPane1, javax.swing.GroupLayout.PREFERRED_SIZE, 439, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(codePanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(jButton5, javax.swing.GroupLayout.PREFERRED_SIZE, 54, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(jButton6, javax.swing.GroupLayout.PREFERRED_SIZE, 54, javax.swing.GroupLayout.PREFERRED_SIZE))
                .addGap(33, 33, 33))
        );

        jPanel1.add(codePanel, "card3");

        javax.swing.GroupLayout jPanel4Layout = new javax.swing.GroupLayout(jPanel4);
        jPanel4.setLayout(jPanel4Layout);
        jPanel4Layout.setHorizontalGroup(
            jPanel4Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 926, Short.MAX_VALUE)
        );
        jPanel4Layout.setVerticalGroup(
            jPanel4Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 622, Short.MAX_VALUE)
        );

        jPanel1.add(jPanel4, "card4");

        javax.swing.GroupLayout jPanel5Layout = new javax.swing.GroupLayout(jPanel5);
        jPanel5.setLayout(jPanel5Layout);
        jPanel5Layout.setHorizontalGroup(
            jPanel5Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 926, Short.MAX_VALUE)
        );
        jPanel5Layout.setVerticalGroup(
            jPanel5Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 622, Short.MAX_VALUE)
        );

        jPanel1.add(jPanel5, "card5");

        javax.swing.GroupLayout jPanel2Layout = new javax.swing.GroupLayout(jPanel2);
        jPanel2.setLayout(jPanel2Layout);
        jPanel2Layout.setHorizontalGroup(
            jPanel2Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 926, Short.MAX_VALUE)
        );
        jPanel2Layout.setVerticalGroup(
            jPanel2Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 622, Short.MAX_VALUE)
        );

        jPanel1.add(jPanel2, "card2");

        javax.swing.GroupLayout layout = new javax.swing.GroupLayout(getContentPane());
        getContentPane().setLayout(layout);
        layout.setHorizontalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addComponent(jPanel1, javax.swing.GroupLayout.Alignment.TRAILING, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
        );
        layout.setVerticalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addComponent(jPanel1, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
        );

        pack();
    }// </editor-fold>//GEN-END:initComponents

    private void jButton1ActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jButton1ActionPerformed
        try {
            acceptableInput();
            
        if(!widthTa.getText().equals("") && !HieghtTa.getText().equals("")){
        removeNodes();
        inputHeight=Integer.parseInt(HieghtTa.getText());
        inputWidth=Integer.parseInt(widthTa.getText());
        generateNods( Integer.parseInt(HieghtTa.getText()), Integer.parseInt(widthTa.getText()));
        setPattern();
        resTa.setText("Results");
       
        }else{
            JOptionPane.showMessageDialog(null, "Unacceptable Width and Height");
        }
        } catch (Exception e) {
            JOptionPane.showMessageDialog(null, e);
        }
    }//GEN-LAST:event_jButton1ActionPerformed

    private void jButton2ActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jButton2ActionPerformed
       removeNodes();
    }//GEN-LAST:event_jButton2ActionPerformed

    private void jButton4ActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jButton4ActionPerformed
generateCode();   panelManager("code");     // TODO add your handling code here:
    }//GEN-LAST:event_jButton4ActionPerformed

    private void manualbActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_manualbActionPerformed
       automatic.setSelected(false);
    }//GEN-LAST:event_manualbActionPerformed

    private void automaticActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_automaticActionPerformed
        manualb.setSelected(false);
    }//GEN-LAST:event_automaticActionPerformed

    private void jButton3ActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jButton3ActionPerformed
        panelManager("code");        // TODO add your handling code here:
    }//GEN-LAST:event_jButton3ActionPerformed

    private void jButton5ActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jButton5ActionPerformed
        panelManager("feature");
    }//GEN-LAST:event_jButton5ActionPerformed

    private void jButton6ActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jButton6ActionPerformed
             //select all function
      codeTA.requestFocusInWindow();
      codeTA.selectAll();
    //copy function    
StringSelection stringSelection = new StringSelection(codeTA.getText());
Clipboard clpbrd = Toolkit.getDefaultToolkit().getSystemClipboard();
clpbrd.setContents(stringSelection, null);
JOptionPane.showMessageDialog(null, "everything has been selected and copied in clipboard");
    }//GEN-LAST:event_jButton6ActionPerformed

    /**
     * @param args the command line arguments
     */
    public static void main(String args[]) {
        /* Set the Nimbus look and feel */
        //<editor-fold defaultstate="collapsed" desc=" Look and feel setting code (optional) ">
        /* If Nimbus (introduced in Java SE 6) is not available, stay with the default look and feel.
         * For details see http://download.oracle.com/javase/tutorial/uiswing/lookandfeel/plaf.html 
         */
        try {
            for (javax.swing.UIManager.LookAndFeelInfo info : javax.swing.UIManager.getInstalledLookAndFeels()) {
                if ("Nimbus".equals(info.getName())) {
                    javax.swing.UIManager.setLookAndFeel(info.getClassName());
                    break;
                }
            }
        } catch (ClassNotFoundException ex) {
            java.util.logging.Logger.getLogger(mainFrame.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (InstantiationException ex) {
            java.util.logging.Logger.getLogger(mainFrame.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (IllegalAccessException ex) {
            java.util.logging.Logger.getLogger(mainFrame.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (javax.swing.UnsupportedLookAndFeelException ex) {
            java.util.logging.Logger.getLogger(mainFrame.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        }
        //</editor-fold>

        /* Create and display the form */
        java.awt.EventQueue.invokeLater(new Runnable() {
            public void run() {
                new mainFrame().setVisible(true);
            }
        });
    }

    // Variables declaration - do not modify//GEN-BEGIN:variables
    private javax.swing.JTextField HieghtTa;
    private javax.swing.JTextField ThicknessTa;
    private javax.swing.JToggleButton automatic;
    private javax.swing.JPanel codePanel;
    private javax.swing.JTextArea codeTA;
    private javax.swing.JPanel featurePanel;
    private javax.swing.JButton jButton1;
    private javax.swing.JButton jButton2;
    private javax.swing.JButton jButton3;
    private javax.swing.JButton jButton4;
    private javax.swing.JButton jButton5;
    private javax.swing.JButton jButton6;
    private javax.swing.JLabel jLabel1;
    private javax.swing.JLabel jLabel2;
    private javax.swing.JLabel jLabel3;
    private javax.swing.JLabel jLabel4;
    private javax.swing.JLabel jLabel5;
    private javax.swing.JPanel jPanel1;
    private javax.swing.JPanel jPanel2;
    private javax.swing.JPanel jPanel4;
    private javax.swing.JPanel jPanel5;
    private javax.swing.JPanel jPanel6;
    private javax.swing.JPanel jPanel7;
    private javax.swing.JScrollPane jScrollPane1;
    private javax.swing.JSeparator jSeparator1;
    private javax.swing.JToggleButton manualb;
    private javax.swing.JComboBox<String> mapCombo;
    private javax.swing.JLabel resTa;
    private javax.swing.JPanel resultPanel;
    private javax.swing.JTextField widthTa;
    // End of variables declaration//GEN-END:variables

    private boolean acceptableInput(){
        
        thickness=Integer.parseInt(ThicknessTa.getText());
        
        return false;
    }
    private void frameSetup() {
       try {
    for (UIManager.LookAndFeelInfo info : UIManager.getInstalledLookAndFeels()) {
        if ("Nimbus".equals(info.getName())) {
            UIManager.setLookAndFeel(info.getClassName());
            break;
        }
    }
} catch (Exception e) {
    // If Nimbus is not available, you can set the GUI to another look and feel.
}
       
       this.nods=new ArrayList<>();
       this.inputHeight=0;
       this.inputWidth=0;
       this.thickness=0;
       
       
        initComponents();
        
        
       widthTa.setText("");
       HieghtTa.setText("");
       widthTa.setHorizontalAlignment(JTextField.CENTER);
       HieghtTa.setHorizontalAlignment(JTextField.CENTER);
       ThicknessTa.setHorizontalAlignment(JTextField.CENTER);
       
        panelManager("feature");
        this.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        this.setLocationRelativeTo(null);
        this.setVisible(true);
    }
    private void panelManager(String panel){
        switch(panel){
            case "feature": codePanel.setVisible(false);featurePanel.setVisible(true);break;
            case "code":featurePanel.setVisible(false); codePanel.setVisible(true);break;
            default: featurePanel.setVisible(true);
        }
    }
}
