// @ hunar ahmad @ Brainxyz

// data & labels
X=[[0,0],[ 1,1], [1,0], [0,1]];  
y=[0, 0, 1, 1];          

// parameters
learn_rate=0.1;
ins=2; 
nodes=4; 
out =1;

// helper functions to generate randomised matrices for intializing weights
function randn(ins, outs){
  W =  []
  for(let i=0; i<ins; i++){
    W.push([])
    for(let j=0; j<outs; j++){
      W[i].push(Math.random()-0.5);
    }
  }
  return W;
}
function zeros(ins, outs){
  W =  []
  for(let i=0; i<ins; i++){
    W.push([])
    for(let j=0; j<outs; j++){
      W[i].push(0);
    }
  }
  return W;
}

// declaring and initializing some arrays
ix=[0,0];
z1=[0,0,0,0];
X2=[0,0,0,0];
deY1=[0,0,0,0];
deX1=[0,0,0,0];
dW2=[0,0,0,0];
W1=randn(ins,nodes);
dW1 = zeros(ins, nodes);
W2=randn(nodes,out);
mse =[];

// training loop
for(let i=0; i<100; i++){
    ers =0;
    for(let j=0; j<X.length; j++){
        
      // feed forward
        ix[0] = X[j][0];
        ix[1] = X[j][1];

        for(let j=0; j<W1[0].length; j++){
          z1[j] = ix[0]*W1[0][j] + ix[1]*W1[0][j];
          X2[j] = Math.sin(z1[j]);
        }

        z2 = X2[0]*W2[0][0] +  X2[1]*W2[1][0] + X2[2]*W2[2][0] + X2[3]*W2[3][0];
        yhat=z2;
        
        // estimate the error
        deX2 = y[j]-yhat; 
        er = deX2;


        // backpropagation of the error deX2..
        for(let j=0; j<W2.length; j++){ 
          deY1[j] = deX2*W2[j][0];
        }
        for(let j=0; j<z1.length; j++){ 
          deX1[j] = deY1[j]*Math.cos(z1[j]);
        }          
        
        for(let j=0; j<X2.length; j++){ 
          dW2[j] = deX2 * X2[j] *learn_rate;
        }
        
        for(let j=0; j<dW2.length; j++){
          W2[j][0] = W2[j][0] + dW2[j];
        }

        for(let j=0; j<ix.length; j++){
          for(let k=0; k<W1[0].length; k++){
            dW1[j][k] = deX1[k] * ix[j] *learn_rate;
          }
        }
        
        for(let j=0; j<ix.length; j++){
          for(let k=0; k<W1[0].length; k++){
            W1[j][k] = W1[j][k] + dW1[j][k];
          }
        }        
        
        ers = ers + Math.abs(er);
        
    }
    mse.push(ers);
}

//print errors
for(let i=0; i<mse.length; i++){
  if(i % 10 ==0){
    console.log(mse[i])
  }
}

// compare the network predictions "yhat" to the true labels "y"
for(let j=0; j<X.length; j++){
        
  ix[0] = X[j][0];
  ix[1] = X[j][1];

  for(let j=0; j<W1[0].length; j++){
    z1[j] = ix[0]*W1[0][j] + ix[1]*W1[0][j];
    X2[j] = Math.sin(z1[j]);
  }

  z2 = X2[0]*W2[0][0] +  X2[1]*W2[1][0] + X2[2]*W2[2][0] + X2[3]*W2[3][0];
  yhat=z2;

  console.log("predictions:")
  console.log(j, "Y:", y[j], "yhat:", yhat)
}
