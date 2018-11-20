package gradient.individualityFactorProvider;

public class ConstantIndividualityFactorProvider implements IIndividualityFactorProvider{

	
	private final double individualityFactor;

	public ConstantIndividualityFactorProvider(double individualityFactor) {
		this.individualityFactor = individualityFactor;
	}
	
	@Override
	public double getIndividualityFactor() {
		return individualityFactor;
	}

}
