import React, { Fragment } from 'react';
import ErrorModal from './ErrorModal';
import NLInput from './NLInput';
import EntityPanel from './EntityPanel';
import CSTPanel from './CSTPanel';
import RegexpPanel from './RegexpPanel';
import { Button, Row, Col, Navbar, NavbarBrand, Container } from 'reactstrap';
import { connect } from 'react-redux';
import { hideApiError, generateModel, showCST, hideCST, showEntities, hideEntities } from '../actions/Actions';

class Home extends React.Component {

  getButtonText = () => {
    if (this.props.entitiesIsLoading)
      return 'Processing Entities...';
    else if (this.props.irIsLoading)
      return 'Processing Sentences...';
    else
      return 'Generate model';
  }

  toggleEntities = () => {
    if (this.props.entitiesIsHidden){
        this.props.onShowEntities();
    }
    else {
        this.props.onHideEntities();
    }
  }

  toggleCST = () => {
    if (this.props.cstIsHidden){
        this.props.onShowCST();
    }
    else {
        this.props.onHideCST();
    }
  }

  render () {
    let self = this;
    return (
      <Fragment>
        <Navbar style={{color: 'white', backgroundColor:'#155E7C'}}>
            <NavbarBrand><h3>Arsenal</h3></NavbarBrand>
        </Navbar>

        <main style={{backgroundColor: '#d0e1e1'}}>
            <Container fluid={true} className="pt-sm-4">
                <ErrorModal 
                        title="Server Error"
                        error={this.props.apiError} 
                        show={this.props.apiError != null}
                        onHide={this.props.onHideApiError}
                />
                <Row className="mb-sm-4">
                    <Col md={12}>
                        <NLInput/>
                    </Col>
                </Row>
                <Row className="mb-sm-4">
                    <Col md={12}>
                        <Button 
                            className="mr-sm-2"
                            disabled={!this.props.nl || this.props.cstIsLoading || this.props.entitiesIsLoading} 
                            onClick={()=>this.props.onGenerateModel(this.props.nl)}>
                                {self.getButtonText()}
                        </Button>
                        <div className="float-right">
                            <Button 
                                disabled={!this.props.entities}
                                className="mr-sm-2"
                                onClick={this.toggleEntities}>
                                    {this.props.entitiesIsHidden?'Show Entities':'Hide Entities'}
                            </Button>                        
                            <Button 
                                disabled={!this.props.cst}
                                onClick={this.toggleCST}>
                                    {this.props.cstIsHidden?'Show CST':'Hide CST'}
                            </Button>
                        </div>
                    </Col>
                </Row>
                {!this.props.entitiesIsHidden && 
                <Row>
                    <Col md={12}>
                        <EntityPanel entities={this.props.entities}/>
                    </Col>
                </Row>}
                {!this.props.cstIsHidden && 
                <Row>
                    <Col md={12}>
                        <CSTPanel cst={this.props.cst}/>
                    </Col>
                </Row>}
                <Row>
                    <Col md={12}>
                        <RegexpPanel title={'Regular Expressions'} value={this.props.regexes}/>
                    </Col>
                </Row>
            </Container>
        </main>

        <footer className="footer text-center" style={{color: 'white', backgroundColor:'#155E7C'}}>
          <div className="container">
            <span>&copy; SRI International 2019</span>
          </div>
        </footer>
      </Fragment>
    );
  }

}

function mapStateToProps(state) {
  return state;
}

function mapDispatchToProps(dispatch,ownProps) {
  return {
      onHideApiError: () => { dispatch(hideApiError()) },
      onGenerateModel: (text) => { dispatch(generateModel(text)) },
      onShowCST: () => { dispatch(showCST()) },
      onHideCST: () => { dispatch(hideCST()) },
      onShowEntities: () => { dispatch(showEntities()) },
      onHideEntities: () => { dispatch(hideEntities()) }
  }
}

export default connect(mapStateToProps,mapDispatchToProps)(Home);
