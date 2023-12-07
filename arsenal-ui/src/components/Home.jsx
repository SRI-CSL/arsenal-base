import React from 'react';
import ErrorModal from './ErrorModal';
import NLInput from './NLInput';
import EntityPanel from './EntityPanel';
import CSTPanel from './CSTPanel';
import ImprovedCSTPanel from './ImprovedCSTPanel';
import { Button, Row, Col, Navbar, NavbarBrand, Container, Label, Input, 
         Nav, NavItem, NavLink, TabContent, TabPane } from 'reactstrap';
import { connect } from 'react-redux';
import { hideApiError, generateModel, setRealEP, setNoOpEP } from '../actions/Actions';
import classnames from 'classnames';

class Home extends React.Component {

  constructor(props) {
    super(props);
    this.state = {
      activeTab: '1'
    };
  }

  selectTab = (tab) => {
    if (this.state.activeTab !== tab) {
      this.setState({
        activeTab: tab
      });
    }
  }

  getButtonText = () => {
    if (this.props.entitiesIsLoading)
      return 'Processing Entities...';
    else if (this.props.cstIsLoading)
      return 'Generating CSTs in Polish notation...';
    else if (this.props.reformulationIsLoading)
      return 'Generating CSTs...';
    else
      return 'Generate CSTs';
  }

  // toggleEntities = () => {
  //   if (this.props.entitiesIsHidden){
  //       this.props.onShowEntities();
  //   }
  //   else {
  //       this.props.onHideEntities();
  //   }
  // }

  toggleProcessEntities = () => {
    if (this.props.noopEntity){
        this.props.setRealEP();
    }
    else {
        this.props.setNoOpEP();
    }
  }

  // toggleCST = () => {
  //   if (this.props.cstIsHidden){
  //       this.props.onShowCST();
  //   }
  //   else {
  //       this.props.onHideCST();
  //   }
  // }

  render () {
    let self = this;
    return (
      <div style={{height: '100vh'}}>
        <ErrorModal 
          title="Server Error"
          error={this.props.apiError} 
          show={this.props.apiError != null}
          onHide={this.props.onHideApiError}/>

        <Navbar style={{color: 'white', backgroundColor:'#155E7C', height: '55px'}}>
            <NavbarBrand><h3>Arsenal</h3></NavbarBrand>
        </Navbar>

        <main style={{backgroundColor: '#d0e1e1', height: 'calc(100vh - 55px)'}} className="mb-sm-4">
          <Container fluid={true}>

            <Nav tabs className="pt-sm-4">
              <NavItem>
                <NavLink
                  className={classnames({ active: self.state.activeTab === '1' })}
                  onClick={() => { self.selectTab('1'); }}
                >
                  English
                </NavLink>
              </NavItem>
              <NavItem>
                <NavLink
                  className={classnames({ active: self.state.activeTab === '2' })}
                  onClick={() => { self.selectTab('2'); }}
                >
                  Entities
                </NavLink>
              </NavItem>        
              <NavItem>
                <NavLink
                  className={classnames({ active: self.state.activeTab === '3' })}
                  onClick={() => { self.selectTab('3'); }}
                >
                  Raw CSTs
                </NavLink>
              </NavItem>        
              <NavItem>
                <NavLink
                  className={classnames({ active: self.state.activeTab === '4' })}
                  onClick={() => { self.selectTab('4'); }}
                >
                  Final CSTs
                </NavLink>
              </NavItem>
            </Nav>      

            <TabContent activeTab={this.state.activeTab}>
              <TabPane tabId="1">
                <Row className="mb-sm-4 mt-sm-4">
                  <Col md={12}>
                    <NLInput/>
                  </Col>
                </Row>
                <Row className="pb-sm-4 ml-sm-2" style={{position: 'fixed', bottom: 0, height: '55px'}}>
                  <Button 
                      className="mr-sm-4"
                      disabled={!this.props.nl || this.props.cstIsLoading || this.props.entitiesIsLoading} 
                      onClick={()=>this.props.onGenerateModel(this.props.nl)}>
                          {self.getButtonText()}
                    </Button>
                    <Label check className="ml-sm-4 mt-sm-2">
                      <Input type="checkbox" 
                              checked={!this.props.noopEntity}
                              onChange={this.toggleProcessEntities}/>{' '}
                      Process entities
                    </Label>
                </Row>
              </TabPane>
              <TabPane tabId="2">
                <Row className="mb-sm-4 mt-sm-4">
                  <Col md={12}>
                    <EntityPanel/>
                  </Col>
                </Row>
              </TabPane>
              <TabPane tabId="3">
                <Row className="mb-sm-4 mt-sm-4">
                  <Col md={12}>
                    <CSTPanel cst={this.props.cst}/>
                  </Col>
                </Row>
              </TabPane>
              <TabPane tabId="4">
                <Row className="mb-sm-4 mt-sm-4">
                  <Col md={12}>
                    <ImprovedCSTPanel cst={this.props.improvedCst}/>
                  </Col>
                </Row>
              </TabPane>
            </TabContent>

          </Container>
        </main>

        {/* <footer className="footer text-center" style={{color: 'white', backgroundColor:'#155E7C'}}>
          <div className="container">
            <span>&copy; SRI International 2021</span>
          </div>
        </footer> */}
      </div>
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
      //onRegenerateModel: (entities) => { dispatch(regenerateModel(entities)) },
      //onShowCST: () => { dispatch(showCST()) },
      //onHideCST: () => { dispatch(hideCST()) },
      //onShowEntities: () => { dispatch(showEntities()) },
      //onHideEntities: () => { dispatch(hideEntities()) },
      setRealEP: () => { dispatch(setRealEP())},
      setNoOpEP: () => { dispatch(setNoOpEP())},
  }
}

export default connect(mapStateToProps,mapDispatchToProps)(Home);
