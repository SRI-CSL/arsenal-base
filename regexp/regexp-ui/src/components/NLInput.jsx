import React from 'react';
import { Card, CardHeader, CardBody } from 'reactstrap';
import { connect } from 'react-redux';
import { setNL } from '../actions/Actions';
import AceEditor from 'react-ace';
import 'brace/theme/github';

class NLInput extends React.Component {
   
    render() {
        return (
            <Card>
                <CardHeader>
                    Specification text
                </CardHeader>
                <CardBody className="pl-0 pr-0 pb-0 pt-0">
                    <AceEditor 
                        name="nl-input-editor"
                        style={{width: '100%'}}
                        fontSize={14}
                        theme="github"
                        showPrintMargin={false}
                        value={this.props.nl} 
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
        onChange: (text) => { dispatch(setNL(text)) }
    }
}

export default connect(mapStateToProps,mapDispatchToProps)(NLInput);
