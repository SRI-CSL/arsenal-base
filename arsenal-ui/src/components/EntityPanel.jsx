import React from 'react';
import { Card, CardBody } from 'reactstrap';
import { connect } from 'react-redux';
import { setEntities } from '../actions/Actions';
import AceEditor from 'react-ace';
import 'brace/theme/github';

class EntityPanel extends React.Component {

    render() {
        return (
            <Card>
                {/* <CardHeader>
                    Entities
                </CardHeader> */}
                <CardBody className="pl-0 pr-0 pb-0 pt-0">
                    <AceEditor 
                        name="entity-panel"
                        mode='json'
                        style={{width: '100%', height: 'calc(100vh - 170px)'}}
                        fontSize={14}
                        theme="github"
                        showPrintMargin={false}
                        value={this.props.entities_string} 
                        onChange={(newText)=>this.props.onChange(newText)}
                    />
                </CardBody>
            </Card>
        );
    }

}

function mapStateToProps(state) {
    return state;
}
  
function mapDispatchToProps(dispatch,ownProps) {
    return {
        onChange: (text) => { dispatch(setEntities(text)) }
    }
}

export default connect(mapStateToProps,mapDispatchToProps)(EntityPanel);
