package gradient.individualityFactorProvider;

import utilities.random.RNGProvider;

public class RandomIndividualityFactorProvider implements IIndividualityFactorProvider{
	
	@Override
	public double getIndividualityFactor() {
		return RNGProvider.getRandom().nextDouble();
	}

}
