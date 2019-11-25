import React from 'react';
import JsonPanel from './JsonPanel';

export default class CSTPanel extends React.Component {

    render() {
        return <JsonPanel name="cst-panel" title="Concrete Syntax Tree (CST)" data={this.props.cst}/>;
    }

}
