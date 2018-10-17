import java.util.*;
import java.math.*;

class TrainingSet{
	
	int[] in;
	
	int out;
	
	boolean done;
	
	TrainingSet(int[] i,int o){
		in = i;
		out = o;
		done = false;
	}
}

class Node{
	int id;
	double in;
	double out;
	double delta;
	Link[] paths;
	
	
	Node(int i){
		id = i;
	}
	
}

class Link{
	
	Node father;
	double weight;
	
	Link (Node f,double w){
		father = f;
		weight = w;
	}
	
	public static void randomInitialize(NeuralNetwork n){

		for(int h = 0; h < n.hidden_n.length ; h++){
			n.hidden_n[h].paths = new Link[n.input_n.length];
			for(int i = 0; i < n.input_n.length ; i++){
				Random r = new Random();
				double random = -1 + r.nextDouble() * 2;
				//double random = r.nextDouble() * 0.1f;
				//double random = r.nextGaussian();
				n.hidden_n[h].paths[i] = new Link(n.input_n[i],random);
				/*int random = r.nextInt(2);
					
				if( random != 1)
					n.hidden_n[h].paths[i] = new Link(n.input_n[i],Double.POSITIVE_INFINITY);
				else
					n.hidden_n[h].paths[i] = new Link(n.input_n[i],Double.NEGATIVE_INFINITY);*/
			}
		}		
		

		for(int o = 0; o < n.output_n.length ; o++){
			n.output_n[o].paths = new Link[n.hidden_n.length];
			for(int h = 0; h < n.hidden_n.length ; h++){
				Random r = new Random();
				double random = -1 + r.nextDouble() * 2;
				//double random = r.nextDouble() * 0.1f;
				//double random = r.nextGaussian();
				
				n.output_n[o].paths[h] = new Link(n.hidden_n[h],random);
				/*int random = r.nextInt(2);
				if( random != 1)
					n.output_n[o].paths[h] = new Link(n.hidden_n[h],Double.POSITIVE_INFINITY);
				else
					n.output_n[o].paths[h] = new Link(n.hidden_n[h],Double.NEGATIVE_INFINITY);*/
					
			}
		}
	}
	
	public static double getInput(Link[] ls) {
		double sum=0;
		for(Link l : ls) {
			sum += l.father.out*l.weight;
			//NeuralNetwork.print(("Sum = " + l.father.out + " * " + l.weight));
		}
		
		return sum;
	}
}


class NeuralNetwork{
	Node[] input_n;
	Link[] paths;
	Node[] hidden_n;
	Node[] output_n;
	
	public static double alpha;
	
	NeuralNetwork(int i,int h,int o){
		input_n = new Node[i];
		for(int c = 0 ; c < i ; c++)
			input_n[c] = new Node(c);
		hidden_n = new Node[h];
		for(int c = 0 ; c < h ; c++)
			hidden_n[c] = new Node(c+i);
		output_n = new Node[o];
		for(int c = 0 ; c < o ; c++)
			output_n[c] = new Node(c+i+h);
		paths = new Link[i*h + h*o];
		Link.randomInitialize(this);
	}
	
	private static double g(double k) {
		double aux = (1/(1+Math.exp(-(double)k))) ;
		/*System.out.println("aux = "+ aux);
		if(aux < 0.5)
			return 0;
		else
			return 1;*/
		return aux;
	}
	
	public static void print(String s) {
		System.out.println(s);
	}
	
	private static double g_dx(double k) {
		double aux =  (Math.exp(-k) / Math.pow((1+Math.exp(-k)),2));
		return aux;
	}
	
	public static double getWeightFromNodeToOut(Node targ,Link[] l) {
		
		for( Link item : l) {
			if(item.father.id == targ.id)
				return item.weight;	
		}
		return 0;
	}
	
	public static double sumWeightDelta(double delta , Link[] l, Node i) {
		double w = getWeightFromNodeToOut(i,l);
			
		return w*delta;
	}
	
	public static double getOutputFromNet(NeuralNetwork net,int[] in) {
		
		for(int i = 0 ; i < net.input_n.length ; i++) 
			net.input_n[i].out = in[i];
		
		for(int i = 0 ; i < net.hidden_n.length ; i++) {
			net.hidden_n[i].in = Link.getInput(net.hidden_n[i].paths);
			net.hidden_n[i].out = g(net.hidden_n[i].in);
		
		}
		
		net.output_n[0].in = Link.getInput(net.output_n[0].paths);
		net.output_n[0].out = g(net.output_n[0].in);
		
		return net.output_n[0].out;
	}
	
	public static NeuralNetwork backPropLearning(TrainingSet[] examples,NeuralNetwork net,int timeLimit) {
		int limit = 800;
		alpha = 0.005f;
		boolean isEnough=false;
		double sum=0;
		long epoch = 0;
		//double delta[] = new double[net.output_n.length+net.hidden_n.length];
		
		long start = System.currentTimeMillis();
		
		Link.randomInitialize(net);
		while(!isEnough) {
			sum = 0;
			for(TrainingSet t : examples) {
				
				/*for( int i = 0 ; i < t.in.length ; i++)
					System.out.print(t.in[i] + " | ");*/
				
				//System.out.println("");
				for(int i = 0 ; i < net.input_n.length ; i++) 
					net.input_n[i].out = t.in[i];
				
				for(int i = 0 ; i < net.hidden_n.length ; i++) {
					net.hidden_n[i].in = Link.getInput(net.hidden_n[i].paths);
					net.hidden_n[i].out = g(net.hidden_n[i].in);
				
				}
				
				net.output_n[0].in = Link.getInput(net.output_n[0].paths);
				net.output_n[0].out = g(net.output_n[0].in);
				
				net.output_n[0].delta = g_dx(net.output_n[0].in) * (t.out - net.output_n[0].out);
				//net.output_n[0].delta = net.output_n[0].out * (1 - net.output_n[0].out) * (t.out - net.output_n[0].out); 
		
				for(Link l : net.output_n[0].paths)
					l.weight = l.weight + alpha * net.output_n[0].out * net.output_n[0].delta;
								
				for(int i = 0 ; i < net.hidden_n.length ; i++) 
					net.hidden_n[i].delta = g_dx(net.hidden_n[i].in) * sumWeightDelta(net.output_n[0].delta,net.output_n[0].paths,net.hidden_n[i]); 
					
				for(int i = 0 ; i < net.hidden_n.length ; i++) {
					for(Link l : net.hidden_n[i].paths)
						l.weight = l.weight + alpha * net.hidden_n[i].out * net.hidden_n[i].delta;
				}
				
				sum += Math.pow(t.out - net.output_n[0].out,2);
				
				/*print("Delta = " + net.output_n[0].delta+"");
				print("Input = " + net.output_n[0].in);
				print("Output = " + net.output_n[0].out);
				System.out.println("");*/
			}
			/*limit--;
			if (limit == 0)
				isEnough = true;
			*/
			//isEnough = true;
			double med = sum/16;
			/*print("		=> Media = " + med);
			print("		=> Number of epochs = " + epoch);
			for(TrainingSet t : examples) {
				if(!t.done)
					isEnough = false;
				else
					t.done = false;
			}*/
			if(med <= 0.1)
				isEnough = true;
			epoch++;
			long elapsedTimeMillis = System.currentTimeMillis()-start;

			// Get elapsed time in seconds
			float elapsedTimeSec = elapsedTimeMillis/1000F;
			if(elapsedTimeSec > timeLimit*60) {
				isEnough = true;
				System.out.println("\nLimite de tempo excedido. Treino terminado.");
			}
		}		
		return net;
	}
	
	public static void renderNeural(NeuralNetwork n) {
		for(Node node : n.hidden_n) {
			for(Link l: node.paths) 
				System.out.println("Caminho entre node " + l.father.id + " e node " + node.id + " inicializado com peso " + l.weight);
		}
		for(Node node : n.output_n) {
			for(Link l: node.paths) 
				System.out.println("Caminho entre node " + l.father.id + " e node " + node.id + " inicializado com peso " + l.weight);
		}		
	}
	
	
}

public class neural {
	
	public static void main (String[] args) {
		
		NeuralNetwork n = new NeuralNetwork(4,16,1);
		
		NeuralNetwork.renderNeural(n);
		
		TrainingSet[] exs = new TrainingSet[16];
		
		int[] in;
		
		/*in = new int[]{1,0,0,0,0,0,0};
		exs[0] = new TrainingSet(in,1);
		
		in = new int[]{1,1,0,0,0,0,0};
		exs[1] = new TrainingSet(in,0);

		in = new int[]{1,1,1,0,0,0,0};
		exs[2] = new TrainingSet(in,1);

		in = new int[]{1,1,1,1,0,0,0};
		exs[3] = new TrainingSet(in,0);
		
		in = new int[]{1,1,1,1,1,0,0};
		exs[4] = new TrainingSet(in,1);

		in = new int[]{1,1,1,1,1,1,0};
		exs[5] = new TrainingSet(in,0);

		in = new int[]{1,1,1,1,1,1,1};
		exs[6] = new TrainingSet(in,1);
		
		in = new int[]{0,1,0,0,0,0,0};
		exs[7] = new TrainingSet(in,1);

		in = new int[]{0,1,1,0,0,0,0};
		exs[8] = new TrainingSet(in,0);*/

		int i1 = 0;
		int i2 = 0;
		int i3 = 0;
		int i4 = 0;
		for( int i = 0 ; i < 16 ; i++) {
			int iCount=0;
			in = new int[4];
			if(i <= 7) {
				in[0] = 1; iCount++;
			}
			
			if((i / 4) % 2 == 0) {
				in[1] = 1; iCount++;
			
			}
			
			if((i / 2) % 2 == 0) {
				in[2] = 1; iCount++;
			}
			
			if(i % 2 == 0) {
				in[3] = 1; iCount++;
			}
			if (iCount % 2 == 1)
				exs[i] = new TrainingSet(in,1);
				else
					exs[i] = new TrainingSet(in,0);
			iCount = 0;
		}
		/*for(TrainingSet t : exs) {
			
			for( int i = 0 ; i < t.in.length ; i++)
				System.out.print(t.in[i] + " | ");
			System.out.println(" - " + t.out);
		}*/
		
		
		System.out.println("\nO treino acabará quando o erro quadrado médio for < 0.1 ou o processo de treino demorar mais de X minutos.");
		System.out.print("\nLimite de tempo (numero de minutos) : ");
		Scanner ins = new Scanner(System.in);
		int t = ins.nextInt();
		
		System.out.println("\n\n	... treino em progresso...");
		/*
		n.output_n[0].delta = 0.23344749934f;
		System.out.println("Sum weight delta");
		System.out.println("Node id numero 5,6,7,8");
		System.out.println("\nID = " + n.hidden_n[0].id);
		System.out.println(NeuralNetwork.sumWeightDelta(n.output_n[0].delta,n.output_n[0].paths,n.hidden_n[0]));
		System.out.println("\nID = " + n.hidden_n[1].id);
		System.out.println(NeuralNetwork.sumWeightDelta(n.output_n[0].delta,n.output_n[0].paths,n.hidden_n[1]));
		System.out.println("\nID = " + n.hidden_n[2].id);
		System.out.println(NeuralNetwork.sumWeightDelta(n.output_n[0].delta,n.output_n[0].paths,n.hidden_n[2]));
		System.out.println("\nID = " + n.hidden_n[3].id);
		System.out.println(NeuralNetwork.sumWeightDelta(n.output_n[0].delta,n.output_n[0].paths,n.hidden_n[3]));*/
		
		NeuralNetwork.backPropLearning(exs, n, t);
		System.out.println("\nTreino completo.");
		
		System.out.println("\nInserir conjunto de input para prever resultado. ( Quatro valores 0 ou 1)");
		in = new int[4];
		for(int i = 0 ; i < 4 ; i++) {
			System.out.print("	=> ");
			in[i] = ins.nextInt();
		}
		double res = NeuralNetwork.getOutputFromNet(n,in);
		if(res > 0.5)
			System.out.println("\nResultado gerado : " + res + "(1)");
		else
			System.out.println("\nResultado gerado : " + res + "(0)");
		ins.close();
	}
}