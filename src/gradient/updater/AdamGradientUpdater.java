package gradient.updater;

import javax.annotation.Nonnegative;
import javax.annotation.Nonnull;

public class AdamGradientUpdater implements IGradientBasedUpdater{

	private final double d1;
	private final double d2;
	
	private final double lr;
	private final @Nonnull double[] s;
	private final @Nonnull double[] r;
	
	private double t = 0;

	public AdamGradientUpdater(@Nonnegative double learningRate, @Nonnegative int variableCount) {
		this(learningRate,0.9,0.99,variableCount);
	}
	
	public AdamGradientUpdater(@Nonnegative double learningRate,@Nonnegative double d1,@Nonnegative double d2, @Nonnegative int variableCount) {
		lr = learningRate;
		this.d1 = d1;
		this.d2 = d2;
		
		s = new double[variableCount];
		r = new double[variableCount];
	}
	
	@Override
	public void update(@Nonnull double[] point,@Nonnull double[] grad, double fintess, double value) {
	    t++;
	    
	    for(int i=0; i<point.length;i++){
	      s[i] = d1*s[i]+(1-d1)*grad[i];
	      r[i] = d2*r[i]+(1-d2)*grad[i]*grad[i];
	      
	      double s_p = s[i]/(1-Math.pow(d1,t));
	      double r_p = r[i]/(1-Math.pow(d2,t));
	      
	      point[i] -= lr*s_p/(Math.sqrt(r_p)+1e-7);
	    }

	}

}
