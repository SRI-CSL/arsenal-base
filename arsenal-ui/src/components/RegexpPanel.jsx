import React from 'react';
import { Card, CardHeader, CardBody } from 'reactstrap';
import AceEditor from 'react-ace';
import 'brace/theme/github';

export default class RegexpPanel extends React.Component {

    // componentDidMount() {
    //     const jsonMode = new SALMode();
    //     this.refs.aceEditor.editor.getSession().setMode(salMode);
    // } session.setMode("ace/mode/js_regex")

    render() {
        let self = this;
        return (
            <div className="mb-sm-4">
                <Card>
                    <CardHeader>
                        {self.props.title}
                    </CardHeader>
                    <CardBody className="pl-0 pr-0 pb-0 pt-0">
                        <AceEditor 
                            ref="aceEditor"
                            name="regexp-editor"
                            mode="text"
                            fontSize={14}
                            style={{width: '100%'}}
                            theme="xcode"
                            value={this.props.value} 
                            //onChange={this.props.onChange} 
                        />
                    </CardBody>
                </Card>
            </div>
        );
    }

}
