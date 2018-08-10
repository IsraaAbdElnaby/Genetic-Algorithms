import java.io.*;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Scanner;

public class Main {

	public static int M;
	public static int L;
	public static int N;
	public static int T;

	public static double[][] Wh;
	public static double[][] Wo;

	final static double learningRate = 0.1;
	final static int limit = 10;


	public static void main(String[] args) throws FileNotFoundException {
		// TODO Auto-generated method stub

	    Scanner in = new Scanner (new File("input.txt"));
	
		M = in.nextInt(); // #of input nodes
		L = in.nextInt(); // #of hidden nodes
		N = in.nextInt(); // #of output nodes
		T = in.nextInt(); // #of training examples
		
		for (int t = 0; t < T; t++) {
			double[] x = new double[M]; // input vector
			double[] y = new double[N]; // output vector

			Wh = new double[L][M]; // input vector
			Wo = new double[N][L]; // output vector

			///////// I/O vectors initialization //////////
			
			for (int i = 0; i < M; i++) {
				x[i] = in.nextDouble();
			}
			for (int i = 0; i < N; i++) {
				y[i] = in.nextDouble();
			}
			

			initialWeights();
			double E = 0.0;  //Mean Square Error
			
			while (true) {
				
	          ////////////////// FEED FORWARD NN /////////////////
				ArrayList<Double> netH = netInputHidden(x); 
				ArrayList<Double> Ij = netOutHidden(netH); 
				ArrayList<Double> netK = netInputOutput(Ij, x);
				ArrayList<Double> Ok = netOutOutput(netK);
				ArrayList<Double> outError = OutputErrors(Ok, y);
				

				E = MSE(outError);

				if(E <= limit) break; //BACK PROPAGATION NOT NEEDED 
				
				//////////////// BACK PROPAGATION //////////////////////////////
				double[][] updatedWo = newOutWeights(outError, Ij);
				ArrayList<Double> hError = HiddenErrors(outError, Ij);
				double[][] updatedWh = newHiddenWeights(hError, x);
				for(int i=0; i<L;i++){
					for(int j=0; j<M; j++){
						Wh[i][j] = updatedWh[i][j];
					}
				}
				for(int i=0; i<N;i++){
					for(int j=0; j<L; j++){
						Wo[i][j] = updatedWo[i][j];
					}
				}
				
				
			}
			System.out.println("MSE = "+ E);
		}

	}

	public static void initialWeights() {
		double rand;
		for (int i = 0; i < L; i++) {
			for (int j = 0; j < M; j++) {
				rand = (double) (-5 + (Math.random() * ((5 - (-5) + 1))));
				Wh[i][j] = rand;

			}
		}

		for (int i = 0; i < N; i++) {
			for (int j = 0; j < L; j++) {
				rand = (float) (-5 + (Math.random() * ((5 - (-5) + 1))));
				Wo[i][j] = rand;
			}
		}
	}

	public static ArrayList<Double> netInputHidden(double[] in) {
		double net = 0.0;
		ArrayList<Double> netJ = new ArrayList<>();
		for (int j = 0; j < L; j++) {
			net = 0.0;
			for (int i = 0; i < M; i++) {
				net += Wh[j][i] * in[i];
			}
			netJ.add(net);
			
		}

		return netJ;
	}

	public static ArrayList<Double> netOutHidden(ArrayList<Double> neti) {
		double f = 0.0;
		ArrayList<Double> I = new ArrayList<>();

		for (int i = 0; i < neti.size(); i++) {
			f = 1 / (1 + Math.exp(neti.get(i) * -1)); // 1 / (1 + e^-netH)
			I.add(f);
		}

		return I;
	}

	public static ArrayList<Double> netInputOutput(ArrayList<Double> I, double[] in) {
		double net = 0.0;
		ArrayList<Double> netK = new ArrayList<>();
		ArrayList<Double> ni = netInputHidden(in);
		I = netOutHidden(ni);

		for (int k = 0; k < N; k++) {
			net = 0.0;
			for (int j = 0; j < L; j++) {
				net += Wo[k][j] * I.get(j);
			}
			netK.add(net);
			
		}

		return netK;
	}

	public static ArrayList<Double> netOutOutput(ArrayList<Double> neto) {
		double f = 0.0;
		ArrayList<Double> O = new ArrayList<>();

		for (int k = 0; k < neto.size(); k++) {
			f = 1 / (1 + Math.exp(neto.get(k) * -1)); // 1 / (1 + e^-netH)
			O.add(f);
		}

		return O;
	}

	//////////////////////// BACK PROPAGATION ///////////////////////
	public static ArrayList<Double> OutputErrors(ArrayList<Double> o, double[] yk) {
		ArrayList<Double> segmaKo = new ArrayList<>();
		double fDash = 0.0, segK = 0.0;
		for (int k = 0; k < N; k++) {
			fDash = o.get(k) * (1 - o.get(k));
			segK = yk[k] - o.get(k);
			segmaKo.add(fDash * segK);
		}
		return segmaKo;
	}

	public static ArrayList<Double> HiddenErrors(ArrayList<Double> oE, ArrayList<Double> I) {
		double sum = 0.0, fDash = 0.0;
		ArrayList<Double> segmaH = new ArrayList<>();
		for (int j = 0; j < L; j++) {
			for (int k = 0; k < N; k++) {
				sum += oE.get(k) * Wo[k][j];
			}
			fDash = I.get(j) * (1 - I.get(j));
			segmaH.add(sum * fDash);
		}

		return segmaH;
	}

	public static double[][] newOutWeights(ArrayList<Double> oE, ArrayList<Double> I) {
		double[][] newWo = new double[N][L];
		for (int k = 0; k < N; k++) {
			for (int j = 0; j < L; j++) {
				newWo[k][j] = Wo[k][j] + (learningRate * oE.get(k) * I.get(j));
			}
		}
		return newWo;
	}

	public static double[][] newHiddenWeights(ArrayList<Double> hE, double[] x) {
		double[][] newWh = new double[L][M];
		for (int j = 0; j < L; j++) {
			for (int i = 0; i < M; i++) {
				newWh[j][i] = Wh[j][i] + (learningRate * hE.get(j) * x[i]);
			}
		}
		return newWh;
	}

	public static double MSE(ArrayList<Double> segmaK) {
		double E = 0.0;
		for (int k = 0; k < N; k++) {
			E += Math.pow(segmaK.get(k), 2.0);
		}
		E /= 2;
		return E;
	}

}
